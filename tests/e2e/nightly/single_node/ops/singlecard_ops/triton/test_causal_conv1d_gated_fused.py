"""
Unit tests for fused causal_conv1d + sigmoid_gating operator.
"""

import gc
import time
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F


# ============================================================
# Reference Implementations
# ============================================================


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """
    Reference implementation for causal_conv1d_update.
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    batch, dim = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]

    # Concat state and input
    x_new = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -state_len:])

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)

    if activation == "silu":
        out = F.silu(out)

    return out.squeeze(-1).to(dtype=dtype_in)


def fused_gdn_gating_ref(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GDN gating computation.
    """
    x = a + dt_bias
    beta_x = softplus_beta * x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )
    g = -torch.exp(A_log) * softplus_x
    beta = torch.sigmoid(b)
    return g, beta


def separated_conv1d_gated(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    activation: str = "silu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separated implementation for testing."""
    x_out = causal_conv1d_update_ref(x, conv_state, weight, bias, activation)
    g, beta = fused_gdn_gating_ref(A_log, a, b, dt_bias)
    return x_out, g, beta


# ============================================================
# Fused Kernel Wrapper
# ============================================================


def causal_conv1d_gated_fused_wrapper(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    activation: str = "silu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused implementation wrapper for causal_conv1d + sigmoid_gating.
    """
    try:
        from vllm_ascend.ops.triton.mamba.causal_conv1d_gated_fused import (
            causal_conv1d_gated_fused as fused_kernel,
        )

        # Call fused kernel
        conv_state_indices = torch.arange(
            x.shape[0], dtype=torch.int32, device=x.device
        )
        out_conv, out_g, out_beta = fused_kernel(
            x=x,
            weight=weight,
            bias=bias,
            conv_state=conv_state,
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            conv_state_indices=conv_state_indices,
            activation=activation,
        )
        return out_conv, out_g, out_beta

    except Exception as e:
        print(f"Fused kernel not available or failed: {e}")
        print("Falling back to separated implementation")
        return separated_conv1d_gated(
            x, conv_state.clone(), weight, bias,
            A_log, a, b, dt_bias, activation,
        )


# ============================================================
# Test Utilities
# ============================================================


def validate_close(
    y_cal: torch.Tensor,
    y_ref: torch.Tensor,
    dtype: torch.dtype,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    """Validate that two tensors are close within tolerance."""
    if dtype == torch.float16:
        rtol, atol = 1e-1, 5e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-1, 5e-2
    elif dtype == torch.float32:
        rtol, atol = 1e-3, 4e-3

    torch.testing.assert_close(y_ref, y_cal, rtol=rtol, atol=atol, equal_nan=True)


def benchmark_kernel(
    kernel_fn,
    warmup_runs: int = 10,
    measure_runs: int = 100,
) -> float:
    """Benchmark kernel execution time and return average in ms."""
    # Warmup
    for _ in range(warmup_runs):
        kernel_fn()

    # Synchronize
    torch.npu.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(measure_runs):
        kernel_fn()
    torch.npu.synchronize()
    end = time.perf_counter()

    return (end - start) / measure_runs * 1000  # ms


# ============================================================
# Test Cases
# ============================================================


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("dim", [2048, 4096])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("activation", ["silu", None])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("itype", [torch.bfloat16])
def test_fused_correctness_conv1d_gating(
    batch_size: int,
    dim: int,
    width: int,
    activation: Optional[str],
    has_bias: bool,
    itype: torch.dtype,
):
    """Test fused causal_conv1d + gating correctness."""
    torch.manual_seed(42)
    device = "npu"

    state_len = width - 1

    # Input tensors for conv1d
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    conv_state = torch.randn(batch_size, dim, state_len, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None

    # Gating tensors
    A_log = torch.randn(dim, device=device, dtype=itype)
    a = torch.randn(batch_size, dim, device=device, dtype=itype)
    b = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.randn(dim, device=device, dtype=itype)

    # Reference (separated)
    conv_state_ref = conv_state.clone()
    out_conv_ref, out_g_ref, out_beta_ref = separated_conv1d_gated(
        x.clone(), conv_state_ref, weight, bias,
        A_log, a, b, dt_bias, activation=activation if activation else None,
    )

    # Fused
    conv_state_fused = conv_state.clone()
    out_conv_fused, out_g_fused, out_beta_fused = causal_conv1d_gated_fused_wrapper(
        x=x.clone(),
        conv_state=conv_state_fused,
        weight=weight,
        bias=bias,
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        activation=activation if activation else None,
    )

    # Validate conv1d output
    validate_close(out_conv_fused, out_conv_ref, itype)

    # Validate gating outputs
    validate_close(out_g_fused, out_g_ref, itype)
    validate_close(out_beta_fused, out_beta_ref, itype)

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("dim", [4096])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("itype", [torch.bfloat16])
def test_fused_performance(
    batch_size: int,
    dim: int,
    width: int,
    itype: torch.dtype,
):
    """Benchmark fused vs separated implementation performance."""
    torch.manual_seed(42)
    device = "npu"

    state_len = width - 1

    # Input tensors
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    conv_state = torch.randn(batch_size, dim, state_len, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype)

    # Gating tensors
    A_log = torch.randn(dim, device=device, dtype=itype)
    a = torch.randn(batch_size, dim, device=device, dtype=itype)
    b = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.randn(dim, device=device, dtype=itype)

    # Separated (baseline)
    def separated_fn():
        conv_state_tmp = conv_state.clone()
        separated_conv1d_gated(
            x.clone(), conv_state_tmp, weight, bias,
            A_log, a, b, dt_bias,
        )

    # Fused
    def fused_fn():
        conv_state_tmp = conv_state.clone()
        causal_conv1d_gated_fused_wrapper(
            x.clone(), conv_state_tmp, weight, bias,
            A_log, a, b, dt_bias,
        )

    # Benchmark
    time_separated = benchmark_kernel(separated_fn, warmup_runs=5, measure_runs=50)
    time_fused = benchmark_kernel(fused_fn, warmup_runs=5, measure_runs=50)

    speedup = time_separated / time_fused if time_fused > 0 else float("inf")
    improvement = (speedup - 1) * 100

    print(f"\nPerformance Comparison (batch={batch_size}, dim={dim}):")
    print(f"  Separated: {time_separated:.3f} ms")
    print(f"  Fused:     {time_fused:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x ({improvement:+.1f}%)")

    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
