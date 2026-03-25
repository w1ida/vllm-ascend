# vLLM Ascend Decoding 性能分析报告

## 测试配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3.5-0.8B |
| 并发度 | max-num-seqs=1 |
| max_model_len | 4096 |
| gpu_memory_utilization | 0.95 |
| 输入 tokens | 5 |
| 输出 tokens | 50 |

## 关键性能指标

- **总 decoding 时间**: 366.8 ms (50 tokens)
- **平均每个 token 延迟**: 7.34 ms/token
- **吞吐量**: 约 136 tokens/s

## 性能瓶颈分析

### 1. 主要耗时算子 (按类别)

| 类别 | 总耗时 (ms) | 占比 | 主要算子 |
|------|-------------|------|----------|
| GEMM/MatMul | 144.93 | 39.51% | matmul, linear, unquantized_gemm |
| Other | 101.39 | 27.64% | 各类杂项算子 |
| Attention | 67.85 | 18.50% | gdn_attention_core, unified_attention |
| Memory Copy | 39.69 | 10.82% | copy_, memcpy, clone |
| GDN (Delta Rule) | 12.33 | 3.36% | ChunkGatedDeltaRuleFunction |

### 2. TOP 10 最耗时算子

| Rank | 算子名称 | 调用次数 | 总耗时 (ms) | 平均 (us) |
|------|----------|----------|-------------|-----------|
| 1 | vllm::gdn_attention_core | 900 | 48.80 | 54.22 |
| 2 | vllm::unquantized_gemm | 164 | 36.17 | 220.55 |
| 3 | aten::copy_ | 4211 | 19.73 | 4.68 |
| 4 | aclnnInplaceCopy | 2799 | 17.68 | 6.32 |
| 5 | aten::clone | 1176 | 16.89 | 14.36 |
| 6 | aten::contiguous | 1050 | 16.68 | 15.89 |
| 7 | ChunkGatedDeltaRuleFunction | 18 | 12.33 | 684.99 |
| 8 | fused_sigmoid_gating_delta_rule | 882 | 11.18 | 12.67 |
| 9 | vllm::unified_attention_with_output | 300 | 6.98 | 23.27 |
| 10 | _causal_conv1d_update_kernel | 882 | 6.29 | 7.14 |

### 3. AI Core 利用率分析

| OP Type | Count | Total Time (us) | Ratio (%) | Core Type |
|---------|-------|-----------------|-----------|-----------|
| MatMulV2 | 5456 | 129489.9 | 55.51% | AI_CORE |
| Transpose | 1008 | 17029.52 | 7.30% | AI_VECTOR |
| Fused Sigmoid | 882 | 11179.24 | 4.79% | AI_VECTOR |
| Conv1d Update | 882 | 6293.20 | 2.70% | AI_VECTOR |
| Layer Norm | 882 | 6266.42 | 2.69% | AI_VECTOR |
| FusedInferAttn | 300 | 6024.70 | 2.58% | MIX_AIC |

## 关键发现

### 1. 主要瓶颈：GEMM/MatMul 占用 39.5% 时间
- 每次 decoding step 需要多次矩阵乘法
- 平均每个 matmul 耗时 220us，相对较高

### 2. 次要瓶颈：Attention 机制占用 18.5% 时间
- gdn_attention_core 被调用 900 次，总耗时 48.8ms
- 平均每次调用 54us

### 3. 内存开销大：Memory Copy 占用 10.8% 时间
- copy_/clone/contiguous 等操作调用次数超过 6000 次
- 存在大量张量内存复制操作

### 4. GDN 特殊算子：Delta Rule 相关 kernel 占用 3.4% 时间
- ChunkGatedDeltaRuleFunction 单次耗时高达 685us
- 这是 Qwen3.5 架构特有的 gating delta 机制

## 优化建议

### 1. 设置 HCCL_OP_EXPANSION_MODE=AIV
```bash
export HCCL_OP_EXPANSION_MODE=AIV
```
- 日志提示：FFTS+ 方法限制了可用 stream 数量
- 设置此环境变量可增加 runtime shapes 支持范围

### 2. 使用更高的并发度
- 当前并发度为 1，GPU 利用率不足
- 建议测试并发度 20+ 以获得更好吞吐量
- 之前测试显示并发度 20 时吞吐量提升 8 倍

### 3. 调整 max_model_len 降低 KV Cache 压力
- 当前设置为 4096，可根据实际需求降低
- 释放更多内存用于计算

### 4. 启用 jemalloc 内存优化 (需要 root 权限)
```bash
apt-get install libjemalloc2
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

### 5. 使用优化的 Python (需要重新安装)
- 安装带 LTO/PGO 优化的 Python 可提升性能
- 参考 `optimization_and_tuning.md` 中的安装指南

## 性能数据文件

Profiling 数据已保存至：
- `/tmp/vllm_profile/rank0_341923_20260321145959955_ascend_pt/`
  - `ASCEND_PROFILER_OUTPUT/operator_details.csv` - 算子级别性能数据
  - `ASCEND_PROFILER_OUTPUT/trace_view.json` - Chrome tracing 格式时间线
  - `ASCEND_PROFILER_OUTPUT/api_statistic.csv` - API 调用统计

## 参考文档

- `/workspace/vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`
- `/workspace/vllm-ascend/docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md`
