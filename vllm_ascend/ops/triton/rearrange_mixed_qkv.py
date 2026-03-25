# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Optimized implementation for rearrange_mixed_qkv operator.
# This replaces einops.rearrange with torch.view to reduce overhead.

from typing import Optional, Tuple

import torch


def rearrange_mixed_qkv_optimized(
    mixed_qkv: Optional[torch.Tensor],
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    tp_size: int = 1,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Optimized version of rearrange_mixed_qkv using torch.view instead of einops.

    Args:
        mixed_qkv: (seq_len, key_dim + key_dim + value_dim) or None
        key_dim: dimension of key projection
        value_dim: dimension of value projection
        head_k_dim: dimension per head for key
        head_v_dim: dimension per head for value
        tp_size: tensor parallel size

    Returns:
        query: (1, seq_len, num_heads_k, head_k_dim)
        key: (1, seq_len, num_heads_k, head_k_dim)
        value: (1, seq_len, num_heads_v, head_v_dim)
    """
    if mixed_qkv is None:
        return None, None, None

    # Split mixed_qkv into query, key, value
    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )

    # Reshape using view (no copy, just stride change)
    # l (h d) -> 1 l h d
    num_heads_k = key_dim // tp_size // head_k_dim
    num_heads_v = value_dim // tp_size // head_v_dim

    query = query.view(1, -1, num_heads_k, head_k_dim)
    key = key.view(1, -1, num_heads_k, head_k_dim)
    value = value.view(1, -1, num_heads_v, head_v_dim)

    # Make contiguous for subsequent operations
    return query.contiguous(), key.contiguous(), value.contiguous()


__all__ = ["rearrange_mixed_qkv_optimized"]
