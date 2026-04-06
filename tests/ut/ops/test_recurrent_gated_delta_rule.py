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

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

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
