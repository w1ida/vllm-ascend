"""
Unit tests for optimized rearrange_mixed_qkv operator.
"""

import pytest
import torch
from einops import rearrange


def rearrange_mixed_qkv_original(mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim, tp_size=1):
    """Original implementation using einops."""
    if mixed_qkv is None:
        return None, None, None

    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )
    query, key = map(
        lambda x: rearrange(x, "l (h d) -> 1 l h d", d=head_k_dim),
        (query, key),
    )
    value = rearrange(value, "l (h d) -> 1 l h d", d=head_v_dim)
    return query.contiguous(), key.contiguous(), value.contiguous()


def rearrange_mixed_qkv_optimized(mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim, tp_size=1):
    """Optimized implementation using torch.view."""
    if mixed_qkv is None:
        return None, None, None

    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )
    num_heads_k = key_dim // tp_size // head_k_dim
    num_heads_v = value_dim // tp_size // head_v_dim
    query = query.view(1, -1, num_heads_k, head_k_dim)
    key = key.view(1, -1, num_heads_k, head_k_dim)
    value = value.view(1, -1, num_heads_v, head_v_dim)
    return query.contiguous(), key.contiguous(), value.contiguous()


@pytest.mark.parametrize("seq_len", [1, 8, 64, 1024])
@pytest.mark.parametrize("key_dim,value_dim", [(512, 512), (1024, 1024)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rearrange_correctness(seq_len, key_dim, value_dim, head_dim, dtype):
    """Test optimized version matches original."""
    torch.manual_seed(42)
    mixed_qkv = torch.randn(seq_len, key_dim + key_dim + value_dim, device="npu", dtype=dtype)

    q_orig, k_orig, v_orig = rearrange_mixed_qkv_original(
        mixed_qkv.clone(), key_dim, value_dim, head_dim, head_dim
    )
    q_opt, k_opt, v_opt = rearrange_mixed_qkv_optimized(
        mixed_qkv.clone(), key_dim, value_dim, head_dim, head_dim
    )

    torch.testing.assert_close(q_orig, q_opt, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_orig, k_opt, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_orig, v_opt, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("seq_len", [1, 1024])
@pytest.mark.parametrize("key_dim,value_dim", [(512, 512)])
@pytest.mark.parametrize("head_dim", [128])
def test_rearrange_performance(seq_len, key_dim, value_dim, head_dim):
    """Benchmark optimized vs original performance."""
    import time

    torch.manual_seed(42)
    mixed_qkv = torch.randn(seq_len, key_dim + key_dim + value_dim, device="npu", dtype=torch.bfloat16)

    warmup = 10
    measure = 100

    # Warmup
    for _ in range(warmup):
        rearrange_mixed_qkv_original(mixed_qkv.clone(), key_dim, value_dim, head_dim, head_dim)
    torch.npu.synchronize()

    # Measure original
    start = time.perf_counter()
    for _ in range(measure):
        rearrange_mixed_qkv_original(mixed_qkv.clone(), key_dim, value_dim, head_dim, head_dim)
    torch.npu.synchronize()
    end = time.perf_counter()
    time_orig = (end - start) / measure * 1000

    # Warmup
    for _ in range(warmup):
        rearrange_mixed_qkv_optimized(mixed_qkv.clone(), key_dim, value_dim, head_dim, head_dim)
    torch.npu.synchronize()

    # Measure optimized
    start = time.perf_counter()
    for _ in range(measure):
        rearrange_mixed_qkv_optimized(mixed_qkv.clone(), key_dim, value_dim, head_dim, head_dim)
    torch.npu.synchronize()
    end = time.perf_counter()
    time_opt = (end - start) / measure * 1000

    speedup = time_orig / time_opt if time_opt > 0 else float("inf")
    improvement = (speedup - 1) * 100

    print(f"\nPerformance Comparison (seq_len={seq_len}):")
    print(f"  Original (einops):  {time_orig:.4f} ms")
    print(f"  Optimized (view):   {time_opt:.4f} ms")
    print(f"  Speedup:            {speedup:.2f}x ({improvement:+.1f}%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
