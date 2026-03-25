"""
Tests for the view-based rearrange_mixed_qkv optimization.
"""

import pytest
import torch
from einops import rearrange

from vllm_ascend.patch.worker.patch_qwen3_5 import AscendQwen3_5GatedDeltaNet
from vllm_ascend.patch.worker.patch_qwen3_next import AscendQwen3Next_GatedDeltaNet


def rearrange_mixed_qkv_reference(
    mixed_qkv: torch.Tensor | None,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    tp_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
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


def make_layer_stub(
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    tp_size: int,
):
    stub = type("LayerStub", (), {})()
    stub.key_dim = key_dim
    stub.value_dim = value_dim
    stub.head_k_dim = head_k_dim
    stub.head_v_dim = head_v_dim
    stub.tp_size = tp_size
    return stub


@pytest.mark.parametrize(
    ("layer_cls", "seq_len", "key_dim", "value_dim", "head_k_dim", "head_v_dim", "tp_size"),
    [
        (AscendQwen3Next_GatedDeltaNet, 1, 512, 512, 128, 128, 1),
        (AscendQwen3Next_GatedDeltaNet, 64, 1024, 1024, 128, 128, 2),
        (AscendQwen3_5GatedDeltaNet, 8, 512, 512, 128, 128, 1),
        (AscendQwen3_5GatedDeltaNet, 128, 1024, 512, 128, 128, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rearrange_mixed_qkv_matches_reference(
    layer_cls,
    seq_len: int,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    tp_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    mixed_qkv = torch.randn(
        seq_len,
        (key_dim + key_dim + value_dim) // tp_size,
        device="npu",
        dtype=dtype,
    )
    layer = make_layer_stub(key_dim, value_dim, head_k_dim, head_v_dim, tp_size)

    expected = rearrange_mixed_qkv_reference(
        mixed_qkv,
        key_dim,
        value_dim,
        head_k_dim,
        head_v_dim,
        tp_size,
    )
    actual = layer_cls.rearrange_mixed_qkv(layer, mixed_qkv)

    for expected_tensor, actual_tensor in zip(expected, actual):
        assert expected_tensor is not None
        assert actual_tensor is not None
        torch.testing.assert_close(expected_tensor, actual_tensor, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "layer_cls",
    [AscendQwen3Next_GatedDeltaNet, AscendQwen3_5GatedDeltaNet],
)
def test_rearrange_mixed_qkv_handles_none(layer_cls):
    layer = make_layer_stub(512, 512, 128, 128, 1)
    assert layer_cls.rearrange_mixed_qkv(layer, None) == (None, None, None)
