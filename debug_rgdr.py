#!/usr/bin/env python3
"""调试 RGDR 算子精度问题"""

import torch
torch.manual_seed(42)

# 测试配置（与测试文件相同）
BATCH, T, NK, NV, DK, DV, S = 2, 2, 4, 4, 16, 16, 4
SCALE = DK ** -0.5
NPU = 'npu'

device = torch.device(NPU)
query = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
key   = torch.randn(T, NK, DK, dtype=torch.bfloat16, device=device)
value = torch.randn(T, NV, DV, dtype=torch.bfloat16, device=device)
beta  = torch.randn(T, NK, dtype=torch.bfloat16, device=device)
state = torch.zeros(S, NK, DK, DV, dtype=torch.bfloat16, device=device)
g     = torch.sigmoid(torch.randn(T, NV, dtype=torch.float32, device=device))

actual_seq_lengths = torch.ones(BATCH, dtype=torch.int32, device=device)
ssm_state_indices  = torch.arange(BATCH, dtype=torch.int32, device=device)

from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import fused_recurrent_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
init_device_properties_triton()

# PyTorch 参考实现
q_ref = query.unsqueeze(0)  # (1, T, NK, DK)
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

# NPU 算子 (根据算子定义，gk 参数不存在)
state_npu = torch.zeros(S, NV, DV, DK, dtype=torch.bfloat16, device=device)  # 注意：NPU 算子期望 (S, NV, DV, DK)
out_npu = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_fp32(
    query=query, key=key, value=value, g=g, beta=beta, state=state_npu,
    scale=float(SCALE), actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=ssm_state_indices, num_accepted_tokens=None,
)

print("=== 输出对比 ===")
print(f"out_ref shape: {out_ref.shape}")  # (1, T, NV, DV)
print(f"out_npu shape: {out_npu.shape}")  # (T, NV, DV)

# 对比输出（第一个 token，第一个 head）
print(f"\nout_ref[0, 0, 0, :5]: {out_ref[0, 0, 0, :5].float().cpu()}")
print(f"out_npu[0, 0, :5]:    {out_npu[0, 0, :5].float().cpu()}")

# 对比状态
print(f"\nstate_ref shape: {state_ref.shape}")  # (S, NV, DK, DV)
print(f"state_npu shape: {state_npu.shape}")  # (S, NV, DV, DK)

# 状态需要转置才能对比
state_ref_transposed = state_ref.transpose(-2, -1)  # (S, NV, DV, DK)
print(f"\nstate_ref[0, 0, 0, :5] (DK, DV): {state_ref[0, 0, 0, :5].float().cpu()}")
print(f"state_npu[0, 0, 0, :5] (DV, DK): {state_npu[0, 0, 0, :5].float().cpu()}")
print(f"state_ref_transposed[0, 0, 0, :5] (DV, DK): {state_ref_transposed[0, 0, 0, :5].float().cpu()}")

# 完整对比
print("\n=== 完整对比 (转置后) ===")
used_idx = ssm_state_indices.tolist()
print(f"used_idx: {used_idx}")

# 对比状态（转置后）
state_ref_used = state_ref_transposed[used_idx].float().cpu()
state_npu_used = state_npu[used_idx].float().cpu()
print(f"\nstate_ref[used_idx] shape: {state_ref_used.shape}")
print(f"state_npu[used_idx] shape: {state_npu_used.shape}")

try:
    torch.testing.assert_close(
        state_npu_used,
        state_ref_used,
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
        msg="转置后状态对比",
    )
    print("状态对比：PASS")
except AssertionError as e:
    print(f"状态对比：FAIL - {e}")
    print(f"最大差异：{(state_npu_used - state_ref_used).abs().max().item()}")

# 对比输出
try:
    torch.testing.assert_close(
        out_npu.float().cpu(),
        out_ref.squeeze(0).float().cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
        msg="输出对比",
    )
    print("输出对比：PASS")
except AssertionError as e:
    print(f"输出对比：FAIL - {e}")
    print(f"最大差异：{(out_npu.float() - out_ref.squeeze(0).float()).abs().max().item()}")
