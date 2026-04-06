#!/usr/bin/env python3
"""验证 MatVecMul 的 bug 假设"""

import torch
torch.manual_seed(42)

# 模拟 MatVecMul 的行为
# stateInUb: (DV, DK) = (16, 16)，每个元素 0.125
# qInUb: (DK,) = (16,)，每个元素 0.125

DV, DK = 16, 16
state = torch.ones(DV, DK, dtype=torch.float32) * 0.125
q = torch.ones(DK, dtype=torch.float32) * 0.125

print("state[0, :5]:", state[0, :5])
print("q[:5]:", q[:5])

# 正确的 MatVecMul 应该是：broadTmp[j, k] = state[j, k] * q[k]
# 即每行的每个元素乘以 q 的对应元素

# 但根据我的分析，当前实现可能是：broadTmp[j, k] = state[j, k] * q[0]
# 即所有元素乘以 q[0]

# 正确的结果
correct_broadTmp = state * q.unsqueeze(0)  # (DV, DK)
print("\n正确的 broadTmp[0, :5]:", correct_broadTmp[0, :5])

# ReduceSum 对 DK 维度求和
correct_out = correct_broadTmp.sum(dim=-1)  # (DV,)
print("正确的输出 out[:5]:", correct_out[:5])
print("正确的输出 out 应该全为:", correct_out[0].item())

# 如果 MatVecMul 的 bug 是所有元素乘以 q[0]
buggy_broadTmp = state * q[0]  # (DV, DK)
print("\nbuggy broadTmp[0, :5]:", buggy_broadTmp[0, :5])

# ReduceSum 对 DK 维度求和
buggy_out = buggy_broadTmp.sum(dim=-1)  # (DV,)
print("buggy 输出 out[:5]:", buggy_out[:5])
print("buggy 输出 out 应该全为:", buggy_out[0].item())

# 但测试结果显示输出全为 0，不是 0.03125
# 所以可能还有另一个问题...

# 让我检查 q 的值是否真的是 0.125
print("\n=== 检查 q 的实际值 ===")
# 在 NPU 算子中，q 从全局内存复制，然后乘以 scale

# 但等等，我刚才的分析可能有问题
# 让我重新看 MatVecMul 的循环结构

# for (uint32_t i = 0; i < alignK_; i += REPEAT_LENTH) {
#     mask = min(REPEAT_LENTH, alignK_ - i)
#     for (uint32_t j = 0; j < cols; j += MAX_REPEAT_TIME) {
#         repeatTime = min(MAX_REPEAT_TIME, cols - j)
#         Mul(dst[j * alignK_ + i], cube[j * alignK_ + i], vec[i], mask, repeatTime, ...)
#     }
# }

# 当 alignK_ = 16, REPEAT_LENTH = 64 时：
# i = 0, mask = 16
# j = 0, repeatTime = 16
# Mul(dst[0], cube[0], vec[0], 16, 16, ...)

# Mul 的参数中，mask=16 表示每次处理 16 个元素
# repeatTime=16 表示重复 16 次

# 所以总共处理 16 * 16 = 256 个元素？
# 但 DST 的大小是 DV * DK = 16 * 16 = 256，这是正确的

# 但问题是 Mul 的语义是什么？
# 如果 Mul(dst[0], cube[0], vec[0], 16, 16, ...) 表示：
# - 每次从 dst[0] 开始取 16 个元素
# - 重复 16 次，每次 stride 是？

# 根据 params {1, 1, 1, repeatStride, repeatStride, 0}
# repeatStride = alignK_ / FP32_NUM_PER_BLOCK = 16 / 8 = 2

# 这意味着每次重复的 stride 是 2 * sizeof(float) = 8 bytes？

# 这很复杂，让我直接看测试结果
print("\n=== 实际测试结果 ===")
print("NPU 输出全为 0，说明问题不在 MatVecMul 的计算逻辑")
print("可能问题在于数据拷贝或队列同步")
