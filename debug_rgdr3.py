#!/usr/bin/env python3
"""调试 RGDR 算子 beta 和 g 的传递"""

import torch
torch.manual_seed(42)

# 测试配置
BATCH, T, NK, NV, DK, DV, S = 2, 2, 4, 4, 16, 16, 4
SCALE = DK ** -0.5
NPU = 'npu'

device = torch.device(NPU)

# 创建更容易追踪的输入数据
query = torch.ones(T, NK, DK, dtype=torch.bfloat16, device=device) * 0.5
key   = torch.ones(T, NK, DK, dtype=torch.bfloat16, device=device) * 0.5
value = torch.ones(T, NV, DV, dtype=torch.bfloat16, device=device) * 0.5
beta  = torch.ones(T, NV, dtype=torch.bfloat16, device=device) * 0.5  # 改为 (T, NV)
g     = torch.zeros(T, NV, dtype=torch.float32, device=device)  # g=0, exp(g)=1

actual_seq_lengths = torch.ones(BATCH, dtype=torch.int32, device=device)
ssm_state_indices  = torch.arange(BATCH, dtype=torch.int32, device=device)

print("=== 输入数据 ===")
print(f"query[0, 0, 0]: {query[0, 0, 0].item()}")
print(f"key[0, 0, 0]:   {key[0, 0, 0].item()}")
print(f"value[0, 0, 0]: {value[0, 0, 0].item()}")
print(f"beta[0, 0]:     {beta[0, 0].item()}")
print(f"g[0, 0]:        {g[0, 0].item()}, exp(g[0,0])={torch.exp(g[0, 0]).item()}")
print(f"scale:          {SCALE}")

from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import fused_recurrent_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
init_device_properties_triton()

# PyTorch 参考实现
q_ref = query.unsqueeze(0)
k_ref = key.unsqueeze(0)
v_ref = value.unsqueeze(0)
g_ref = g.unsqueeze(0)
beta_ref = beta.unsqueeze(0)
initial_state = torch.zeros(S, NV, DK, DV, dtype=torch.bfloat16, device=device)
cu_seqlens = torch.arange(BATCH + 1, dtype=torch.long, device=device)

out_ref, state_ref = fused_recurrent_gated_delta_rule_pytorch(
    q=q_ref, k=k_ref, v=v_ref, g=g_ref, beta=beta_ref,
    initial_state=initial_state, inplace_final_state=True,
    cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices.to(torch.long),
    num_accepted_tokens=None,
)

print("\n=== PyTorch 参考实现输出 ===")
# 手动计算第一个 token
# h = 0
# h = h * exp(g) = 0
# delta = v - sum(h * k) = v = 0.5
# delta = delta * beta = 0.5 * 0.5 = 0.25
# h = h + delta * k = 0 + 0.25 * 0.5 = 0.125
# o = sum(h * q) = sum(0.125 * 0.5) = 0.125 * 0.5 * DK = 0.125 * 0.5 * 16 = 1.0

# 等等，让我重新计算
# h: (NV, DV, DK)
# q: (DK,) -> (NV, 1, DK) 扩展后
# k: (DK,) -> (NV, 1, DK) 扩展后
# v: (DV,)

# 对于 head 0:
# h[0]: (DV, DK) 全 0
# q[0]: (DK,) = 0.5
# k[0]: (DK,) = 0.5
# v[0]: (DV,) = 0.5

# h = h * exp(g) = 0
# Sk = sum(h * k, dim=-1) = 0 (因为 h=0)
# delta = v - Sk = 0.5
# delta = delta * beta = 0.5 * 0.5 = 0.25
# h = h + delta.unsqueeze(-1) * k.unsqueeze(0)
#   = 0 + 0.25.unsqueeze(-1) * 0.5.unsqueeze(0)
#   = (DV, 1) * (1, DK) = (DV, DK) 全 0.125
# o = sum(h * q, dim=-1)
#   = sum(0.125 * 0.5, dim=-1) = 0.125 * 0.5 * DK = 0.125 * 0.5 * 16 = 1.0

print(f"out_ref[0, 0, 0, 0]: {out_ref[0, 0, 0, 0].item()} (预期约 1.0)")
print(f"out_ref[0, 0, 0, :5]: {out_ref[0, 0, 0, :5].tolist()}")
print(f"state_ref[0, 0, 0, :5]: {state_ref[0, 0, 0, :5].tolist()}")

# NPU 算子
state_npu = torch.zeros(S, NV, DV, DK, dtype=torch.bfloat16, device=device)
out_npu = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_fp32(
    query=query, key=key, value=value, g=g, beta=beta, state=state_npu,
    scale=float(SCALE), actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=ssm_state_indices, num_accepted_tokens=None,
)

print("\n=== NPU 算子输出 ===")
print(f"out_npu[0, 0, 0]: {out_npu[0, 0, 0].item()} (预期约 1.0)")
print(f"out_npu[0, 0, :5]: {out_npu[0, 0, :5].tolist()}")
print(f"state_npu[0, 0, 0, :5]: {state_npu[0, 0, 0, :5].tolist()}")

print("\n=== 对比 ===")
print(f"输出差异：{(out_npu - out_ref.squeeze(0).to(torch.bfloat16)).abs().max().item()}")
print(f"状态差异：{(state_npu - state_ref.transpose(-2, -1).to(torch.bfloat16)).abs().max().item()}")
