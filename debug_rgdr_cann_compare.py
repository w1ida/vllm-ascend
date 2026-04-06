#!/usr/bin/env python3
"""对比 CANN 原始算子与我们的算子实现"""
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# 测试配置
BATCH, T, NK, NV, DK, DV, S = 2, 2, 4, 4, 16, 16, 4
SCALE = DK ** -0.5
device = 'npu'

# 构造输入 - 使用 CANN 算子期望的布局
torch.manual_seed(42)
query = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
key = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
value = torch.randn(T, NV, DV, dtype=torch.bfloat16, device=device)
g = torch.sigmoid(torch.randn(T, NV, dtype=torch.float32, device=device))
beta = torch.randn(T, NV, dtype=torch.bfloat16, device=device)  # (T, NV)

# CANN 算子期望 state 布局：(S, NV, DV, DK)
state_cann = torch.zeros(S, NV, DV, DK, dtype=torch.bfloat16, device=device)
state_ours = torch.zeros(S, NV, DV, DK, dtype=torch.bfloat16, device=device)

actual_seq_lengths = torch.ones(BATCH, dtype=torch.int32, device=device)
ssm_state_indices = torch.arange(BATCH, dtype=torch.int32, device=device)

print("=== 输入数据 ===")
print(f"query shape: {query.shape}")  # (T, NK, DK)
print(f"key shape: {key.shape}")      # (T, NK, DK)
print(f"value shape: {value.shape}")  # (T, NV, DV)
print(f"g shape: {g.shape}")          # (T, NV)
print(f"beta shape: {beta.shape}")    # (T, NV)
print(f"state_cann shape: {state_cann.shape}")  # (S, NV, DV, DK)
print(f"state_ours shape: {state_ours.shape}")  # (S, NV, DV, DK)

# 1. 调用 CANN 原始算子 (bf16 state)
print("\n=== 调用 CANN 原始算子 ===")
try:
    cann_out = torch_npu.npu_recurrent_gated_delta_rule(
        query=query, key=key, value=value, state=state_cann,
        beta=beta, g=g, scale=SCALE,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
    )
    print(f"CANN 算子调用成功")
    print(f"CANN out shape: {cann_out.shape}")
    print(f"CANN out[0,0,:5]: {cann_out[0,0,:5].float().cpu()}")
    print(f"CANN out 是否全零：{(cann_out == 0).all().item()}")
except Exception as e:
    print(f"CANN 算子调用失败：{e}")
    cann_out = None

# 2. 调用我们的算子 (支持 fp32 state，但也支持 bf16)
print("\n=== 调用我们的算子 ===")
try:
    our_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_fp32(
        query=query, key=key, value=value, state=state_ours,
        beta=beta, g=g, scale=SCALE,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
    )
    print(f"我们的算子调用成功")
    print(f"Our out shape: {our_out.shape}")
    print(f"Our out[0,0,:5]: {our_out[0,0,:5].float().cpu()}")
    print(f"Our out 是否全零：{(our_out == 0).all().item()}")
except Exception as e:
    print(f"我们的算子调用失败：{e}")
    our_out = None

# 3. 对比输出
if cann_out is not None and our_out is not None:
    print("\n=== 输出对比 ===")
    print(f"CANN out[0,0,:5]:  {cann_out[0,0,:5].float().cpu()}")
    print(f"Our out[0,0,:5]:   {our_out[0,0,:5].float().cpu()}")
    print(f"输出差异 max: {(cann_out - our_out).abs().max().item()}")

    # 4. 对比状态
    print("\n=== 状态对比 ===")
    print(f"CANN state[0,0,0,:5]:  {state_cann[0,0,0,:5].float().cpu()}")
    print(f"Our state[0,0,0,:5]:   {state_ours[0,0,0,:5].float().cpu()}")
    print(f"状态差异 max: {(state_cann - state_ours).abs().max().item()}")
else:
    print("\n=== 无法对比 ===")
    print("其中一个算子调用失败")
