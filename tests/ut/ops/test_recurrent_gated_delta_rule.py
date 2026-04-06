#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for torch.ops._C_ascend.npu_recurrent_gated_delta_rule.

Operator shapes (from tiling / infershape sources):
  query  : (T, NK, DK)   bf16
  key    : (T, NK, DK)   bf16
  value  : (T, NV, DV)   bf16
  beta   : (T, NK)       bf16
  state  : (S, NK, DK, DV)  bf16  -- mutated in-place
  actual_seq_lengths : (B,)  int32
  ssm_state_indices  : (B,)  int32
  g      : (T, NV)       float32  [optional]
  num_accepted_tokens: (B,)  int32  [optional]
  scale  : float  (passed as Python float; maps to the scale_value attr)

Constraints (from tiling source):
  0 < NK <= 256,  0 < NV <= 256,  NV % NK == 0
  0 < DK <= 512,  0 < DV <= 512
  S >= T  (state pool must be at least as large as the token count)
"""

import time

import pytest
import torch

from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import (
    fused_recurrent_gated_delta_rule_pytorch,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
init_device_properties_triton()

# ---------------------------------------------------------------------------
# Try to use the triton kernel; fall back to the PyTorch reference when triton
# is not functional on this backend (e.g. Ascend NPU without CUDA triton).
# The PyTorch reference is the ground-truth implementation that the triton
# kernel is designed to match, so precision results are equivalent.
# ---------------------------------------------------------------------------
try:
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule as _triton_fused

    def _triton_reference(q, k, v, g, beta, scale, initial_state,
                          inplace_final_state, cu_seqlens,
                          ssm_state_indices, num_accepted_tokens):
        return _triton_fused(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    # Probe call to detect runtime failures (e.g. missing triton.next_power_of_2)
    _probe_q = torch.zeros(1, 1, 1, 16, dtype=torch.bfloat16, device="npu")
    _probe_v = torch.zeros(1, 1, 1, 16, dtype=torch.bfloat16, device="npu")
    _probe_g = torch.zeros(1, 1, 1, dtype=torch.float32, device="npu")
    _probe_b = torch.zeros(1, 1, 1, dtype=torch.bfloat16, device="npu")
    _probe_s = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16, device="npu")
    _triton_reference(
        q=_probe_q, k=_probe_q, v=_probe_v, g=_probe_g, beta=_probe_b,
        scale=1.0, initial_state=_probe_s, inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.long, device="npu"),
        ssm_state_indices=torch.tensor([0], dtype=torch.long, device="npu"),
        num_accepted_tokens=None,
    )
    _REFERENCE_LABEL = "triton"
except Exception:
    # Triton kernel not available; use the PyTorch reference (functionally identical)
    def _triton_reference(q, k, v, g, beta, scale, initial_state,  # type: ignore[misc]
                          inplace_final_state, cu_seqlens,
                          ssm_state_indices, num_accepted_tokens):
        return fused_recurrent_gated_delta_rule_pytorch(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    _REFERENCE_LABEL = "pytorch_ref (triton unavailable)"

# ---------------------------------------------------------------------------
# Shared fixture: small but valid dimensions
# ---------------------------------------------------------------------------
BATCH = 2       # B – number of sequences in the batch
T = BATCH       # T – total tokens; in decode mode every sequence has 1 token
NK = 4          # number of query/key heads
NV = 4          # number of value heads  (must be multiple of NK)
DK = 16         # key head dimension
DV = 16         # value head dimension
S = 4           # state pool slots (must be >= T)
SCALE = DK ** -0.5

NPU = "npu"


def _make_inputs(state_dtype: torch.dtype):
    """Build a dict of inputs ready to pass to the operator."""
    device = torch.device(NPU)

    query = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
    key   = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
    value = torch.randn(T, NV, DV, dtype=torch.bfloat16, device=device)
    beta  = torch.randn(T, NK, dtype=torch.bfloat16, device=device)
    state = torch.zeros(S, NK, DK, DV, dtype=state_dtype, device=device)
    g     = torch.sigmoid(torch.randn(T, NV, dtype=torch.float32, device=device))

    # Decode: each of the B sequences contributes exactly 1 token
    actual_seq_lengths = torch.ones(BATCH, dtype=torch.int32, device=device)
    # Map each sequence to a distinct state slot (slots 0 … B-1)
    ssm_state_indices  = torch.arange(BATCH, dtype=torch.int32, device=device)

    return dict(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        state=state,
        scale=float(SCALE),
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecurrentGatedDeltaRuleOutputShape:
    """Output tensor must have the same shape as `value` and be bf16."""

    def test_bf16_state_output_shape_and_dtype(self):
        inputs = _make_inputs(torch.bfloat16)
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        assert out.shape == (T, NV, DV), f"Expected shape {(T, NV, DV)}, got {out.shape}"
        assert out.dtype == torch.bfloat16, f"Expected bf16 output, got {out.dtype}"

    def test_fp32_state_output_shape_and_dtype(self):
        inputs = _make_inputs(torch.float32)
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        assert out.shape == (T, NV, DV), f"Expected shape {(T, NV, DV)}, got {out.shape}"
        assert out.dtype == torch.bfloat16, f"Expected bf16 output, got {out.dtype}"


class TestRecurrentGatedDeltaRuleStateMutation:
    """The `state` tensor must be modified in-place by the operator."""

    def test_bf16_state_is_mutated(self):
        inputs = _make_inputs(torch.bfloat16)
        state_before = inputs["state"].clone()
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        assert not torch.equal(inputs["state"], state_before), \
            "state tensor should have been updated in-place (bf16)"

    def test_fp32_state_is_mutated(self):
        inputs = _make_inputs(torch.float32)
        state_before = inputs["state"].clone()
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        assert not torch.equal(inputs["state"], state_before), \
            "state tensor should have been updated in-place (fp32)"


class TestRecurrentGatedDeltaRuleStateSlotIndexing:
    """Only the state slots referenced by ssm_state_indices should change."""

    def test_unindexed_state_slots_unchanged(self):
        inputs = _make_inputs(torch.bfloat16)
        # Sequences use slots 0 and 1; slots 2 and 3 should remain zero
        state_ref = inputs["state"].clone()
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        updated_state = inputs["state"]

        # Slots that were written (indices 0 and 1) should have changed
        used_idx = inputs["ssm_state_indices"].tolist()
        assert not torch.equal(updated_state[used_idx], state_ref[used_idx]), \
            "Used state slots should be updated"

        # Slots that were NOT written should remain zero
        unused = [i for i in range(S) if i not in used_idx]
        if unused:
            assert torch.equal(updated_state[unused], state_ref[unused]), \
                "Unused state slots should not be modified"


class TestRecurrentGatedDeltaRuleBatchDecode:
    """Larger batch with distinct per-sequence state slots."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_variable_batch_sizes(self, batch_size):
        device = torch.device(NPU)
        t = batch_size  # decode: 1 token per sequence

        query = torch.randn(t, NK, DK, dtype=torch.bfloat16, device=device)
        key   = torch.randn(t, NK, DK, dtype=torch.bfloat16, device=device)
        value = torch.randn(t, NV, DV, dtype=torch.bfloat16, device=device)
        beta  = torch.randn(t, NK, dtype=torch.bfloat16, device=device)
        state = torch.zeros(batch_size + 2, NK, DK, DV, dtype=torch.bfloat16, device=device)
        g     = torch.sigmoid(torch.randn(t, NV, dtype=torch.float32, device=device))
        actual_seq_lengths = torch.ones(batch_size, dtype=torch.int32, device=device)
        ssm_state_indices  = torch.arange(batch_size, dtype=torch.int32, device=device)

        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            state=state,
            scale=float(SCALE),
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=None,
        )
        assert out.shape == (t, NV, DV)
        assert out.dtype == torch.bfloat16


class TestRecurrentGatedDeltaRuleOptionalParams:
    """Optional parameters: num_accepted_tokens."""

    def test_with_num_accepted_tokens(self):
        """Pass num_accepted_tokens as an int32 tensor (speculative-decode scenario)."""
        device = torch.device(NPU)
        inputs = _make_inputs(torch.bfloat16)
        # All sequences accepted exactly 1 token
        inputs["num_accepted_tokens"] = torch.ones(BATCH, dtype=torch.int32, device=device)
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        assert out.shape == (T, NV, DV)
        assert out.dtype == torch.bfloat16

    def test_without_g(self):
        """g is OPTIONAL according to the op definition; passing None should work."""
        inputs = _make_inputs(torch.bfloat16)
        inputs["g"] = None
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        assert out.shape == (T, NV, DV)
        assert out.dtype == torch.bfloat16


class TestRecurrentGatedDeltaRuleNumerical:
    """Numerical sanity checks for the operator output."""

    def test_outputs_are_finite(self):
        """Output should contain no NaN or Inf values."""
        inputs = _make_inputs(torch.bfloat16)
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        out_f32 = out.float()
        assert torch.isfinite(out_f32).all(), \
            "Non-finite values in output"

    def test_fp32_state_outputs_are_finite(self):
        """fp32 state: output should contain no NaN or Inf values."""
        inputs = _make_inputs(torch.float32)
        out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        out_f32 = out.float()
        assert torch.isfinite(out_f32).all(), \
            "Non-finite values in output (fp32 state)"


# ---------------------------------------------------------------------------
# Helper: convert NPU op flat-token inputs to triton varlen-batch format
# ---------------------------------------------------------------------------

def _npu_to_triton_inputs(inputs: dict) -> dict:
    """Convert _make_inputs() dict to kwargs for fused_recurrent_gated_delta_rule.

    NPU op uses a flat token layout (T, heads, dim) with per-sequence
    metadata tensors.  The triton function uses a batched layout (B, T, heads,
    dim) with variable-length sequence descriptors (cu_seqlens).  For decode
    mode every sequence contributes exactly one token so the mapping is:
        cu_seqlens = [0, 1, 2, …, B]   (each slice has length 1)
    """
    device = torch.device(NPU)
    B = int(inputs["actual_seq_lengths"].shape[0])
    S = inputs["state"].shape[0]
    NV_dim = inputs["value"].shape[1]   # NV  (value heads)
    DK_dim = inputs["query"].shape[2]   # DK
    DV_dim = inputs["value"].shape[2]   # DV

    g = inputs.get("g")
    nat = inputs.get("num_accepted_tokens")

    # Initial state for triton: (S, NV, DK, DV) all zeros (same dtype as NPU state)
    initial_state = torch.zeros(
        S, NV_dim, DK_dim, DV_dim,
        dtype=inputs["state"].dtype,
        device=device,
    )

    # cu_seqlens: [0, 1, …, B] — decode mode, 1 token per sequence
    cu_seqlens = torch.arange(B + 1, dtype=torch.long, device=device)

    return dict(
        q=inputs["query"].unsqueeze(0),           # (1, T, NK, DK)
        k=inputs["key"].unsqueeze(0),             # (1, T, NK, DK)
        v=inputs["value"].unsqueeze(0),           # (1, T, NV, DV)
        g=g.unsqueeze(0) if g is not None else None,  # (1, T, NV)
        beta=inputs["beta"].unsqueeze(0),         # (1, T, NK)
        scale=inputs["scale"],
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=inputs["ssm_state_indices"].to(torch.long),  # (B,)
        num_accepted_tokens=nat,
    )


def _npu_sync() -> None:
    """Synchronize the NPU device (best-effort)."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Precision comparison: NPU op vs triton
# ---------------------------------------------------------------------------

class TestRecurrentGatedDeltaRuleVsTriton:
    """Compare the NPU custom op against the triton/PyTorch reference."""

    def test_output_matches_triton(self):
        """NPU op output must be numerically close to the reference output."""
        torch.manual_seed(42)
        inputs = _make_inputs(torch.bfloat16)

        # --- NPU op ---
        npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)

        # --- triton / pytorch-reference ---
        tk = _npu_to_triton_inputs(inputs)
        ref_out, _ = _triton_reference(**tk)
        ref_out = ref_out.squeeze(0)  # (1,T,NV,DV) → (T,NV,DV)

        torch.testing.assert_close(
            npu_out.float().cpu(),
            ref_out.float().cpu(),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=True,
            msg=f"NPU op output diverges from {_REFERENCE_LABEL} reference",
        )

    def test_state_matches_triton(self):
        """Updated state slots must be numerically close to the reference state.

        Comparison is valid when NK == NV so both implementations store a
        tensor of shape (S, NK/NV, DK, DV) per state slot.
        """
        assert NK == NV, "State comparison requires NK == NV"

        torch.manual_seed(42)
        inputs = _make_inputs(torch.bfloat16)
        tk = _npu_to_triton_inputs(inputs)

        # --- NPU op (state mutated in-place) ---
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        npu_state = inputs["state"].clone()  # snapshot after update

        # Reset both states to zeros, then re-run for a clean reference
        inputs["state"].zero_()
        tk["initial_state"].zero_()
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        _, ref_state = _triton_reference(**tk)
        # ref_state: (S, NV, DK, DV)

        used_idx = inputs["ssm_state_indices"].tolist()
        torch.testing.assert_close(
            npu_state[used_idx].float().cpu(),
            ref_state[used_idx].float().cpu(),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=True,
            msg=f"NPU op state diverges from {_REFERENCE_LABEL} reference",
        )


# ---------------------------------------------------------------------------
# Performance comparison: NPU op vs triton
# ---------------------------------------------------------------------------

class TestRecurrentGatedDeltaRulePerformance:
    """Latency benchmark: NPU custom op vs triton/PyTorch reference."""

    N_WARMUP = 5
    N_ITERS = 20

    def test_performance_vs_triton(self):
        """Report per-iteration latency for both implementations."""
        inputs = _make_inputs(torch.bfloat16)
        tk = _npu_to_triton_inputs(inputs)

        # --- warm-up ---
        for _ in range(self.N_WARMUP):
            torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        for _ in range(self.N_WARMUP):
            _triton_reference(**tk)

        # --- time NPU op ---
        _npu_sync()
        t0 = time.perf_counter()
        for _ in range(self.N_ITERS):
            torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**inputs)
        _npu_sync()
        npu_ms = (time.perf_counter() - t0) * 1000 / self.N_ITERS

        # --- time reference ---
        _npu_sync()
        t0 = time.perf_counter()
        for _ in range(self.N_ITERS):
            _triton_reference(**tk)
        _npu_sync()
        ref_ms = (time.perf_counter() - t0) * 1000 / self.N_ITERS

        speedup = ref_ms / npu_ms
        print(
            f"\n[perf] 参考实现={_REFERENCE_LABEL} | "
            f"T={T}, NK={NK}, NV={NV}, DK={DK}, DV={DV} | "
            f"NPU算子: {npu_ms:.3f} ms | 参考: {ref_ms:.3f} ms | "
            f"加速比(参考/NPU): {speedup:.2f}x"
        )

        # Sanity guard: NPU op must not be pathologically slower
        assert npu_ms < ref_ms * 100, (
            f"NPU op ({npu_ms:.3f} ms) is >100x slower than reference "
            f"({ref_ms:.3f} ms) — investigate regression"
        )
