#!/usr/bin/env python3
"""手动模拟 NPU 算子的计算逻辑"""

import torch
torch.manual_seed(42)

# 测试配置
BATCH, T, NK, NV, DK, DV, S = 2, 2, 4, 4, 16, 16, 4
SCALE = DK ** -0.5
NPU = 'npu'

device = torch.device(NPU)

# 创建简单输入
query = torch.ones(T, NK, DK, dtype=torch.bfloat16, device=device) * 0.5
key   = torch.ones(T, NK, DK, dtype=torch.bfloat16, device=device) * 0.5
value = torch.ones(T, NV, DV, dtype=torch.bfloat16, device=device) * 0.5
beta  = torch.ones(T, NV, dtype=torch.bfloat16, device=device) * 0.5
g     = torch.zeros(T, NV, dtype=torch.float32, device=device)

actual_seq_lengths = torch.ones(BATCH, dtype=torch.int32, device=device)
ssm_state_indices  = torch.arange(BATCH, dtype=torch.int32, device=device)

print("=== 手动模拟 NPU 算子计算 ===")
print(f"query shape: {query.shape}")  # (T, NK, DK)
print(f"key shape: {key.shape}")      # (T, NK, DK)
print(f"value shape: {value.shape}")  # (T, NV, DV)
print(f"beta shape: {beta.shape}")    # (T, NV)
print(f"g shape: {g.shape}")          # (T, NV)

# NPU 算子的计算逻辑（根据 Compute 函数）
# 状态布局：(S, NV, DV, DK)
state = torch.zeros(S, NV, DV, DK, dtype=torch.float32, device=device)

# 第一个 batch (seq0=0, seq1=1), head_i=0
seq0, seq1, head_i = 0, 1, 0
stateOffset = ssm_state_indices[seq0].item()  # 0

print(f"\n=== 第一个 token, head 0 ===")
print(f"stateOffset: {stateOffset}")

# CopyInQKV
# qOffset = (seq0 * NK + head_i / (NV / NK)) * DK = 0
# vOffset = (seq0 * NV + head_i) * DV = 0
q = query[seq0, head_i].float()  # (DK,)
k = key[seq0, head_i].float()    # (DK,)
v = value[seq0, head_i].float()  # (DV,)

print(f"q[:5]: {q[:5]}")
print(f"k[:5]: {k[:5]}")
print(f"v[:5]: {v[:5]}")

# q = q * scale
q = q * SCALE
print(f"q * scale[:5]: {q[:5]}")

# CopyInState
# curStateOffset = ((stateOffset * NV + head_i) * DV + v_i) * DK = 0
s = state[stateOffset, head_i, :, :].float()  # (DV, DK)
print(f"state[0, 0, 0, :5]: {s[0, :5]}")

# Compute
# alpha = exp(g) = exp(0) = 1
# state = state * alpha = state (因为 alpha=1)
alpha = torch.exp(g[seq0, head_i])
print(f"alpha: {alpha}")

# state = state (不变，因为初始为 0)

# tmp = state * k (元素乘，broadcast)
# state: (DV, DK), k: (DK,)
# tmp[i, j] = state[i, j] * k[j]
tmp = s * k.unsqueeze(0)  # (DV, DK)
print(f"tmp[0, :5]: {tmp[0, :5]}")

# delta = sum(tmp, dim=-1) 对 DK 维度求和
# 根据 ReduceSum<Pattern::Reduce::AR>，AR 可能表示 Across Row
# 如果 stateShape = {DV, DK}，AR 是对每行求和，结果是 (DV,)
delta = tmp.sum(dim=-1)  # (DV,)
print(f"delta[:5]: {delta[:5]}")

# delta = v - delta
delta = v - delta  # (DV,)
print(f"v - delta[:5]: {delta[:5]}")

# delta = delta * beta
beta_val = beta[seq0, head_i]
delta = delta * beta_val
print(f"delta * beta[:5]: {delta[:5]}")

# broadTmp = delta (broadcast 到 (DV, DK))
# state = state + broadTmp * k
# broadTmp: (DV, DK), k: (DK,)
# broadTmp * k: (DV, DK)
# state += broadTmp * k
broadTmp = delta.unsqueeze(-1)  # (DV, 1)
s = s + broadTmp * k.unsqueeze(0)  # (DV, DK)
print(f"state[0, :5] after update: {s[0, :5]}")

# 输出计算
# tmp = state * q
tmp = s * q.unsqueeze(0)  # (DV, DK)
print(f"tmp for output[0, :5]: {tmp[0, :5]}")

# out = sum(tmp, dim=-1) 对 DK 维度求和
out = tmp.sum(dim=-1)  # (DV,)
print(f"out[:5]: {out[:5]}")

print("\n=== 对比 ===")
print(f"手动计算的 out[:5]: {out[:5]}")
print(f"预期值：每个元素应该是 0.125 * 0.5 * 16 = 1.0? 不对...")

# 让我重新计算
# s = delta * k^T = 0.25 * 0.5 = 0.125 (所有元素)
# out = sum(s * q, dim=-1) = sum(0.125 * 0.25, dim=-1) = 0.125 * 0.25 * 16 = 0.5
# 不对，q = 0.5 * 0.25 = 0.125
# out = sum(0.125 * 0.125, dim=-1) = 0.125 * 0.125 * 16 = 0.25

print(f"\n重新计算:")
print(f"delta = v * beta = 0.5 * 0.5 = 0.25")
print(f"s = delta * k^T = 0.25 * 0.5 = 0.125")
print(f"q_scaled = q * scale = 0.5 * 0.25 = 0.125")
print(f"out = sum(s * q_scaled, dim=-1) = sum(0.125 * 0.125) = 0.125 * 0.125 * 16 = {0.125 * 0.125 * 16}")
