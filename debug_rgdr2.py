#!/usr/bin/env python3
"""调试 RGDR 算子输出问题"""

import torch
torch.manual_seed(42)

# 测试配置
BATCH, T, NK, NV, DK, DV, S = 2, 2, 4, 4, 16, 16, 4
SCALE = DK ** -0.5
NPU = 'npu'

device = torch.device(NPU)
query = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
key   = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
value = torch.randn(T, NV, DV, dtype=torch.bfloat16, device=device)
beta  = torch.randn(T, NK, dtype=torch.bfloat16, device=device)
g     = torch.sigmoid(torch.randn(T, NV, dtype=torch.float32, device=device))

actual_seq_lengths = torch.ones(BATCH, dtype=torch.int32, device=device)
ssm_state_indices  = torch.arange(BATCH, dtype=torch.int32, device=device)

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

# NPU 算子 - 使用正确的状态布局 (S, NV, DV, DK)
state_npu = torch.zeros(S, NV, DV, DK, dtype=torch.bfloat16, device=device)
out_npu = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_fp32(
    query=query, key=key, value=value, g=g, beta=beta, state=state_npu,
    scale=float(SCALE), actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=ssm_state_indices, num_accepted_tokens=None,
)

print("=== 第一个 token 的详细分析 ===")
print(f"query[0, 0, :5]: {query[0, 0, :5].float().cpu()}")
print(f"key[0, 0, :5]:   {key[0, 0, :5].float().cpu()}")
print(f"value[0, 0, :5]: {value[0, 0, :5].float().cpu()}")
print(f"g[0, 0]:         {g[0, 0].cpu()}")
print(f"beta[0, 0]:      {beta[0, 0].float().cpu()}")

# 手动计算第一个 token 的输出
print("\n=== 手动计算第一个 token (head=0) ===")
q0 = query[0, 0].float() * SCALE  # (DK,)
k0 = key[0, 0].float()            # (DK,)
v0 = value[0, 0].float()          # (DV,)
g0 = torch.exp(g[0, 0].float())   # scalar
beta0 = beta[0, 0].float()        # scalar

print(f"q0_scaled[:5]: {q0[:5].cpu()}")
print(f"k0[:5]:        {k0[:5].cpu()}")
print(f"v0[:5]:        {v0[:5].cpu()}")
print(f"g0:            {g0.cpu()}")
print(f"beta0:         {beta0.cpu()}")

# 初始状态 (NV, DV, DK)
h = torch.zeros(NV, DV, DK, dtype=torch.float32, device=device)

# t=0 计算
# h = h * g0 (但 h=0 所以还是 0)
# delta = v0 - sum(h * k0, dim=-1) = v0 (因为 h=0)
# delta = delta * beta0
# h = h + delta.unsqueeze(-1) * k0.unsqueeze(0).unsqueeze(0)
# 这里 delta 是 (NV, DV), k0 是 (DK,)
# delta.unsqueeze(-1): (NV, DV, 1)
# k0.unsqueeze(0).unsqueeze(0): (1, 1, DK)
# 结果：(NV, DV, DK)

delta = v0 * beta0  # (NV,) 因为 v0 是 (DV,) 而 beta0 是标量
print(f"delta[:5]: {delta[:5].cpu()}")

# h = delta.unsqueeze(-1) * k0.unsqueeze(0).unsqueeze(-1)  # (NV, 1, DK) -> (NV, DV, DK)
# 等等，v0 是 (DV,) 不是 (NV, DV)
# 让我重新考虑...

# 实际上 v0 是 (NV, DV) 因为 value 的形状是 (T, NV, DV)
# 但在这个测试中 NV=DV=16，所以 v0 是 (16,)
# 不对，value[0, 0] 的形状是 (DV,) = (16,)
# value[0] 的形状是 (NV, DV) = (4, 16)

# 让我重新看输入
print(f"\nv0 shape: {v0.shape}")  # 应该是 (DV,) = (16,)
print(f"value shape: {value.shape}")  # (T, NV, DV) = (2, 4, 16)

# 哦！问题在这里：
# query: (T, NK, DK) = (2, 4, 16)
# key: (T, NK, DK) = (2, 4, 16)
# value: (T, NV, DV) = (2, 4, 16)

# 所以 query[0, 0] 是 (DK,) = (16,)
# value[0, 0] 是 (DV,) = (16,)

# 但在参考实现中，对于每个 head 都要计算
# head_i 从 0 到 NV-1
# q_hv = _expand_to_hv(q_t, HV)  # 将 q 从 (K,) 扩展到 (HV, K)
# 但这里 NK=NV=4，所以每个 head 对应一个 query head

# 实际上，当 NK=NV 时，每个 value head 对应一个 query/key head
# 所以 head_i=0 时，使用 q[0] 和 k[0]

# 让我重新计算
print("\n=== 重新计算（考虑 head 维度）===")
# head_i = 0
q0_head0 = query[0, 0].float() * SCALE  # (DK,) = (16,)
k0_head0 = key[0, 0].float()            # (DK,) = (16,)
v0_head0 = value[0, 0].float()          # (DV,) = (16,)
g0_head0 = torch.exp(g[0, 0].float())   # scalar
beta0_head0 = beta[0, 0].float()        # scalar

# 初始状态 (NV, DV, DK) 但每个 head 有自己的状态
# 对于 head 0，状态是 h[0] 形状 (DV, DK)
h_head0 = torch.zeros(DV, DK, dtype=torch.float32, device=device)

# t=0 计算
# h = h * g0 = 0
# delta = v0 - sum(h * k, dim=-1) = v0 (因为 h=0)
# 这里 sum(h * k, dim=-1) 中 h 是 (DV, DK), k 是 (DK,)
# k 扩展成 (1, DK), h * k 是 (DV, DK)
# sum(..., dim=-1) 对 DK 维度求和，结果是 (DV,)

# 等等，参考实现的逻辑是：
# h_t: (HV, V, K)
# k_hv: (HV, K)
# k_hv.unsqueeze(-2): (HV, 1, K)
# h_t * k_hv.unsqueeze(-2): (HV, V, K)
# sum(..., dim=-1): (HV, V)

# 对于单个 head (HV=1):
# h: (V, K), k: (K,)
# k.unsqueeze(-2): (1, K)
# h * k.unsqueeze(-2): (V, K)
# sum(..., dim=-1): (V,)

delta_head0 = v0_head0  # (DV,) = (16,)
delta_scaled_head0 = delta_head0 * beta0_head0  # (DV,)
print(f"delta_scaled_head0[:5]: {delta_scaled_head0[:5].cpu()}")

# h = 0 + delta_scaled.unsqueeze(-1) * k.unsqueeze(0)
# delta_scaled.unsqueeze(-1): (DV, 1)
# k.unsqueeze(0): (1, DK)
# 结果：(DV, DK)
h_head0 = delta_scaled_head0.unsqueeze(-1) * k0_head0.unsqueeze(0)
print(f"h_head0[0, :5]: {h_head0[0, :5].cpu()}")

# 输出 o = sum(h * q, dim=-1)
# h: (DV, DK), q: (DK,)
# q.unsqueeze(0): (1, DK)
# h * q.unsqueeze(0): (DV, DK)
# sum(..., dim=-1): (DV,)
o_head0 = torch.sum(h_head0 * q0_head0.unsqueeze(0), dim=-1)  # (DV,)
print(f"o_head0[:5]: {o_head0[:5].cpu()}")

# 但输出应该是 (NV, DV) = (4, 16)
# 所以需要对所有 head 计算
print("\n=== 计算所有 head ===")
outputs = []
for head_i in range(NV):
    q_h = query[0, head_i].float() * SCALE  # (DK,)
    k_h = key[0, head_i].float()            # (DK,)
    v_h = value[0, head_i].float()          # (DV,)
    g_h = torch.exp(g[0, head_i].float())   # scalar
    beta_h = beta[0, head_i].float()        # scalar

    h_h = torch.zeros(DV, DK, dtype=torch.float32, device=device)
    delta_h = v_h * beta_h  # (DV,)
    h_h = delta_h.unsqueeze(-1) * k_h.unsqueeze(0)  # (DV, DK)
    o_h = torch.sum(h_h * q_h.unsqueeze(0), dim=-1)  # (DV,)
    outputs.append(o_h)

outputs = torch.stack(outputs)  # (NV, DV)
print(f"outputs[0, :5]: {outputs[0, :5].cpu()}")
print(f"outputs shape: {outputs.shape}")

print(f"\nout_ref[0, 0, :, :5]:\n{out_ref[0, 0, :, :5].float().cpu()}")
print(f"outputs vs out_ref:")
print(f"  outputs[0, :5]: {outputs[0, :5].cpu()}")
print(f"  out_ref[0, 0, 0, :5]: {out_ref[0, 0, 0, :5].float().cpu()}")

print("\n=== NPU 算子的输出 ===")
print(f"out_npu[0, 0, :5]: {out_npu[0, 0, :5].float().cpu()}")
print(f"out_npu[0, 1, :5]: {out_npu[0, 1, :5].float().cpu()}")
print(f"out_npu[0, 2, :5]: {out_npu[0, 2, :5].float().cpu()}")
print(f"out_npu[0, 3, :5]: {out_npu[0, 3, :5].float().cpu()}")

print(f"\nout_ref[0, 0, 0, :5]: {out_ref[0, 0, 0, :5].float().cpu()}")
print(f"out_ref[0, 0, 1, :5]: {out_ref[0, 0, 1, :5].float().cpu()}")
print(f"out_ref[0, 0, 2, :5]: {out_ref[0, 0, 2, :5].float().cpu()}")
print(f"out_ref[0, 0, 3, :5]: {out_ref[0, 0, 3, :5].float().cpu()}")
