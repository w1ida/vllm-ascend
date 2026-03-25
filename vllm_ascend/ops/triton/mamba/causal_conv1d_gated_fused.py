# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused kernel for causal_conv1d_update + sigmoid_gating.
# This combines Mamba convolution with GDN gating in a single kernel
# to reduce global memory access and improve performance.
#
# NOTE: This is a simplified version that focuses on conv1d + gating fusion.
# The full fusion with delta rule is more complex and will be added later.

import os
from typing import Optional, Tuple

import torch
from vllm.triton_utils import tl, tldevice, triton


# ============================================================
# Fast math operations
# ============================================================

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
else:
    exp = tl.exp
    log = tl.log


# ============================================================
# Fused Kernel: Causal Conv1D + Sigmoid Gating
# ============================================================


@triton.jit(do_not_specialize=["num_tokens", "D"])
def causal_conv1d_gated_fused_kernel(
    # Causal conv1d inputs
    x_ptr,  # (num_tokens, D)
    weight_ptr,  # (D, width)
    bias_ptr,  # (D,) or None
    conv_state_ptr,  # (num_states, D, state_len)
    conv_state_indices_ptr,  # (num_tokens,) int32

    # Gating inputs
    A_log_ptr,  # (D,)
    a_ptr,  # (num_tokens, D)
    b_ptr,  # (num_tokens, D)
    dt_bias_ptr,  # (D,)

    # Outputs
    out_conv_ptr,  # (num_tokens, D) - conv1d output
    out_g_ptr,  # (num_tokens, D) - gating output g
    out_beta_ptr,  # (num_tokens, D) - gating output beta

    # Dimensions
    num_tokens: tl.constexpr,
    D: tl.constexpr,
    width: tl.constexpr,
    state_len: tl.constexpr,

    # Strides
    stride_x_token: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_w_d: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_d: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_d: tl.constexpr,

    # Metaprogramming constants
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for causal_conv1d_update + sigmoid_gating.

    Program IDs:
    - pid_token: token index
    - pid_d_block: channel block

    This kernel:
    1. Loads input x and conv state
    2. Applies causal conv1d
    3. Computes sigmoid gating (g, beta)
    4. Stores outputs
    """
    pid_token = tl.program_id(0)
    pid_d_block = tl.program_id(1)

    # Channel indices for this program
    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # ========== Load conv1d weights ==========
    weight_base = weight_ptr + d_offsets * stride_w_d

    # Unroll weight loading for width=4
    w0 = tl.load(weight_base + 0 * stride_w_width, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(weight_base + 1 * stride_w_width, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(weight_base + 2 * stride_w_width, mask=mask_d, other=0.0).to(tl.float32)
    w3 = tl.load(weight_base + 3 * stride_w_width, mask=mask_d, other=0.0).to(tl.float32)

    # Load bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + d_offsets, mask=mask_d, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Load A_log and dt_bias for gating
    A_log = tl.load(A_log_ptr + d_offsets, mask=mask_d, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + d_offsets, mask=mask_d, other=0.0).to(tl.float32)

    # ========== Get state index ==========
    state_idx = tl.load(conv_state_indices_ptr + pid_token)

    # ========== Load input x ==========
    x_offset = pid_token * stride_x_token + d_offsets * stride_x_d
    x_val = tl.load(x_ptr + x_offset, mask=mask_d, other=0.0).to(tl.float32)

    # ========== Load conv state history ==========
    # Reference implementation:
    #   x_new = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
    #   out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)
    # For width=4, state_len=3:
    #   x_new = [state[0], state[1], state[2], x]
    #   out = state[0]*w[0] + state[1]*w[1] + state[2]*w[2] + x*w[3] + bias

    conv_state_base = conv_state_ptr + state_idx * stride_conv_state_seq + d_offsets * stride_conv_state_d

    # Load state history: state[0], state[1], state[2] for width=4, state_len=3
    state0_ptr = conv_state_base + 0 * stride_conv_state_tok
    state1_ptr = conv_state_base + 1 * stride_conv_state_tok
    state2_ptr = conv_state_base + 2 * stride_conv_state_tok

    state0 = tl.load(state0_ptr, mask=mask_d, other=0.0).to(tl.float32)
    state1 = tl.load(state1_ptr, mask=mask_d, other=0.0).to(tl.float32)
    state2 = tl.load(state2_ptr, mask=mask_d, other=0.0).to(tl.float32)
    # state3 is the current input x
    state3 = x_val

    # ========== Causal Conv1D: y = sum(x_new[t-i] * w[i]) + bias ==========
    # x_new = [state0, state1, state2, x] corresponds to [t-3, t-2, t-1, t]
    # Apply weight[:, 0] to state0, weight[:, 1] to state1, etc.
    acc = bias + state0 * w0 + state1 * w1 + state2 * w2 + state3 * w3

    # Apply SiLU activation
    if SILU_ACTIVATION:
        acc = acc / (1.0 + exp(-acc))

    # Store conv1d output
    out_conv_offset = pid_token * stride_x_token + d_offsets * stride_x_d
    tl.store(out_conv_ptr + out_conv_offset, acc, mask=mask_d)

    # ========== Sigmoid Gating ==========

    # Load a and b for gating
    a_offset = pid_token * stride_a_token + d_offsets * stride_a_d
    b_offset = pid_token * stride_a_token + d_offsets * stride_a_d

    a_val = tl.load(a_ptr + a_offset, mask=mask_d, other=0.0).to(tl.float32)
    b_val = tl.load(b_ptr + b_offset, mask=mask_d, other=0.0).to(tl.float32)

    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    x_gating = a_val + dt_bias
    # softplus with numerical stability
    softplus_val = tl.where(
        x_gating <= 20.0,
        log(1.0 + exp(x_gating)),
        x_gating,
    )
    g = -exp(A_log) * softplus_val

    # Compute beta = sigmoid(b)
    beta = 1.0 / (1.0 + exp(-b_val))

    # Store gating outputs
    out_g_offset = pid_token * stride_a_token + d_offsets * stride_a_d
    tl.store(out_g_ptr + out_g_offset, g, mask=mask_d)
    tl.store(out_beta_ptr + out_g_offset, beta, mask=mask_d)


def causal_conv1d_gated_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state_indices: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused implementation of causal_conv1d_update + sigmoid_gating.

    Args:
        x: (num_tokens, D) - input for conv1d
        weight: (D, width) - conv1d weight
        bias: (D,) or None - conv1d bias
        conv_state: (num_states, D, state_len) - conv state buffer
        A_log: (D,) - log(A) for gating
        a: (num_tokens, D) - input a for gating
        b: (num_tokens, D) - input b for sigmoid
        dt_bias: (D,) - dt bias
        conv_state_indices: (num_tokens,) - state indices
        activation: "silu" or None

    Returns:
        out_conv: (num_tokens, D) - conv1d output
        out_g: (num_tokens, D) - gating output g
        out_beta: (num_tokens, D) - gating output beta
    """
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    conv_state = conv_state.contiguous()
    A_log = A_log.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    dt_bias = dt_bias.contiguous()

    # Get dimensions
    num_tokens, D = x.shape
    _, width = weight.shape
    state_len = conv_state.shape[-1]
    num_states = conv_state.shape[0]

    # Allocate outputs
    out_conv = torch.empty_like(x)
    out_g = torch.empty((num_tokens, D), dtype=x.dtype, device=x.device)
    out_beta = torch.empty((num_tokens, D), dtype=x.dtype, device=x.device)

    if conv_state_indices is None:
        conv_state_indices = torch.arange(num_tokens, dtype=torch.int32, device=x.device)
    else:
        conv_state_indices = conv_state_indices.contiguous()

    # Metaprogramming constants
    BLOCK_D = min(512, triton.next_power_of_2(D))

    # Grid
    grid = (num_tokens, triton.cdiv(D, BLOCK_D))

    # Launch kernel
    causal_conv1d_gated_fused_kernel[grid](
        x, weight, bias, conv_state, conv_state_indices,
        A_log, a, b, dt_bias,
        out_conv, out_g, out_beta,
        num_tokens=num_tokens,
        D=D,
        width=width,
        state_len=state_len,
        stride_x_token=x.stride(0),
        stride_x_d=x.stride(1),
        stride_w_d=weight.stride(0),
        stride_w_width=weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_d=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_a_token=a.stride(0),
        stride_a_d=a.stride(1),
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        BLOCK_D=BLOCK_D,
    )

    return out_conv, out_g, out_beta


__all__ = ["causal_conv1d_gated_fused", "causal_conv1d_gated_fused_kernel"]
