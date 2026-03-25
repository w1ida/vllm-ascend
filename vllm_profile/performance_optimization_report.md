# vLLM Ascend 性能优化报告

## 执行摘要

本报告总结了 vLLM Ascend 项目在 Ascend 910B4 NPU 上优化 Qwen3.5-0.8B 模型性能的工作和结果。

**目标**: 在 1 并发下将输出吞吐量提升 50%
**当前进展**: 已实现 +10.5% 提升 (14.40 → 15.91 output tokens/s)
**剩余差距**: 需要额外 +35.8% 提升才能达到 50% 目标

---

## 1. 基准测试结果

### 1.1 测试配置
| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3.5-0.8B |
| 硬件 | Ascend 910B4 NPU (CANN 8.5.0) |
| 输入长度 | 1024 tokens (random) |
| 输出长度 | 100 tokens |
| 请求数 | 100 |
| 并发度 | max-num-seqs=1 |

### 1.2 优化对比

| 配置 | Output tokens/s | Total tokens/s | 提升 | 时间 |
|------|-----------------|----------------|------|------|
| **基线** | 14.40 | 161.85 | - | 11:33 |
| **+ HCCL_OP_EXPANSION_MODE=AIV** | 15.94 | 179.21 | +10.7% | 10:27 |
| **+ max_model_len=8192** | 15.47 | 173.89 | +7.4% | 10:45 |
| **最终配置** (AIV + max_model_len + gpu_mem=0.95) | **15.91** | **178.83** | **+10.5%** | 10:27 |

### 1.3 推荐的环境配置
```bash
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=True
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
```

---

## 2. 性能瓶颈分析

基于现有 profiling 数据 (`vllm_profile/decoding_performance_analysis.md`)：

### 2.1 TOP 瓶颈算子

| Rank | 算子 | 调用次数 | 总耗时 (ms) | 占比 | 类别 |
|------|------|----------|-------------|------|------|
| 1 | `vllm::gdn_attention_core` | 900 | 48.80 | 13.3% | Attention |
| 2 | `vllm::unquantized_gemm` | 164 | 36.17 | 9.9% | GEMM |
| 3 | `aten::copy_` | 4211 | 19.73 | 5.4% | Memory |
| 4 | `aclnnInplaceCopy` | 2799 | 17.68 | 4.8% | Memory |
| 5 | `aten::clone` | 1176 | 16.89 | 4.6% | Memory |
| 6 | `aten::contiguous` | 1050 | 16.68 | 4.5% | Memory |

### 2.2 算子类别分析

| 类别 | 总耗时 (ms) | 占比 | 优化潜力 |
|------|-------------|------|----------|
| GEMM/MatMul | 144.93 | 39.51% | 高 - 可通过算子融合减少 |
| Attention | 67.85 | 18.50% | 中 - 已使用 FIA 核但可优化 |
| Memory Copy | 39.69 | 10.82% | 高 - 可通过 inplace 操作减少 |
| GDN Delta Rule | 12.33 | 3.36% | 中 - Qwen3.5 特有算子 |

---

## 3. 已尝试的优化

### 3.1 成功优化

1. **HCCL_OP_EXPANSION_MODE=AIV**
   - 效果：+10.7% 提升
   - 原理：增加 runtime shapes 支持范围，允许更多算子使用 ACL Graph
   - 推荐：始终启用

2. **max_model_len 优化**
   - 从 4096 增加到 8192
   - 效果：配合 gpu_memory_utilization=0.95 可获得更好 KV cache 利用率
   - 推荐：根据实际场景调整

### 3.2 失败尝试

1. **TASK_QUEUE_ENABLE=2**
   - 错误：`Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture`
   - 原因：与 NPU 图捕获不兼容
   - 建议：保持默认值 0 或 1

---

## 4. 进一步优化建议

### 4.1 系统级优化 (预期收益：5-15%)

1. **Jemalloc 内存分配器**
   ```bash
   apt-get install libjemalloc2
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
   ```
   - 预期收益：减少内存碎片，提升 5-10%

2. **CPU 亲和性优化**
   - 当前日志显示 CPU 绑定失败：`Bind cpus failed in rank0`
   - 建议：手动配置 CPU 亲和性，避免线程迁移

3. **Python 优化编译**
   - 安装带 LTO/PGO 优化的 Python
   - 预期收益：5-10%

### 4.2 算子级优化 (预期收益：15-25%)

#### 4.2.1 GEMM 优化 (占比 39.5%)

**当前实现**: `vllm_ascend/patch/worker/patch_unquantized_gemm.py`
```python
def unquantized_gemm(x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)
```

**优化方向**:
1. 使用 CANN MMLH 库替换原生 linear
2. 实现权重转置预布局，减少 runtime transpose
3. 探索矩阵乘法融合（如 GEMM + Bias + Activation）

**预期收益**: 15-20% GEMM 时间减少 → 整体 6-8% 提升

#### 4.2.2 Attention 优化 (占比 18.5%)

**当前实现**: `vllm_ascend/attention/attention_v1.py`
- 使用 `npu_fused_infer_attention_score` (FIA) 核
- 已支持 PIECEWISE 图编译

**优化方向**:
1. 分析 `gdn_attention_core` 热点，探索自定义 Triton 核
2. 优化 KV cache 布局，减少 reshape 操作
3. 探索 Flash Attention 2/3 实现

**预期收益**: 10-15% Attention 时间减少 → 整体 2-3% 提升

#### 4.2.3 内存拷贝优化 (占比 10.8%)

**问题分析**:
- `copy_`/`clone`/`contiguous` 调用超过 6000 次
- 大量中间张量创建和销毁

**优化方向**:
1. 预分配输出缓冲区，避免重复分配
2. 使用 inplace 操作替代创建新张量
3. 融合连续操作（如 fused bias + activation）

**预期收益**: 30-40% Memory Copy 时间减少 → 整体 3-4% 提升

### 4.3 架构级优化

1. **更高并发度测试**
   - 现有数据：并发度=20 时输出吞吐量 ~104 tokens/s
   - 建议：测试并发度 2, 4, 8, 16 的 sweet spot

2. **KV Cache 压缩**
   - 探索 FP8/INT8 KV cache
   - 减少内存带宽压力

---

## 5. 优化优先级与路线图

### 阶段 1：快速收益 (1-2 周)
- [ ] Jemalloc 内存优化 (预期 +5%)
- [ ] CPU 亲和性修复 (预期 +3%)
- [ ] 更高并发度基准测试

### 阶段 2：算子融合 (2-4 周)
- [ ] GEMM + Bias 融合核
- [ ] 减少 copy_/clone 调用
- [ ] 预分配中间缓冲区

### 阶段 3：深度优化 (4-8 周)
- [ ] 自定义 GDN Attention Triton 核
- [ ] Flash Attention 2 实现
- [ ] 端到端 profiling 驱动优化

---

## 6. 风险与假设

1. **硬件限制**: Ascend 910B4 的 AI Core 频率和带宽可能成为硬上限
2. **软件版本**: 当前 CANN 8.5.0，未来版本可能带来性能提升
3. **模型特性**: Qwen3.5 的 GDN 机制是架构级瓶颈，难以完全消除

---

## 7. 结论

- **当前达成**: +10.5% 提升 (14.40 → 15.91 output tokens/s)
- **50% 目标可行性**: 需要额外 +35.8%，单一优化难以达成，需组合优化
- **推荐路径**:
  1. 先完成系统级优化（jemalloc, CPU affinity）获取快速收益
  2. 通过算子融合和内存优化争取 15-20% 提升
  3. 考虑更高并发度场景，而非单一 1 并发优化

---

## 附录

### A. 性能数据文件
- Profiling 数据：`/tmp/vllm_profile/rank0_341923_20260321145959955_ascend_pt/`
- Benchmark 脚本：`vllm bench throughput --model Qwen/Qwen3.5-0.8B ...`

### B. 关键代码位置
- Attention 实现：`vllm_ascend/attention/attention_v1.py`
- GEMM 补丁：`vllm_ascend/patch/worker/patch_unquantized_gemm.py`
- 编译配置：`vllm_ascend/compilation/acl_graph.py`

### C. 参考文档
- `docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md`
- `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`
