# vLLM Ascend 性能优化总结

## 优化前后对比

### 1 并发吞吐量测试 (Qwen3.5-0.8B, 1024+100 tokens)

| 阶段 | 配置 | 输出吞吐量 (tokens/s) | 总吞吐量 (tokens/s) | 提升 |
|------|------|----------------------|--------------------|------|
| **基线** | 无优化 | 14.40 | 161.85 | - |
| **优化后** | HCCL_OP_EXPANSION_MODE=AIV + max_model_len=8192 + gpu_mem=0.95 | **15.91** | **178.83** | **+10.5%** |

### 算子性能分析

基于 profiling 数据的关键瓶颈算子：

| 算子 | 耗时 (ms) | 调用次数 | 优化方向 |
|------|-----------|----------|----------|
| vllm::gdn_attention_core | 48.80 | 900 | 自定义 Triton 核 |
| vllm::unquantized_gemm | 36.17 | 164 | CANN MMLH 融合 |
| aten::copy_ | 19.73 | 4211 | 预分配缓冲区 |
| aten::clone | 16.89 | 1176 | Inplace 操作 |

---

## 已应用的优化

### 1. HCCL_OP_EXPANSION_MODE=AIV
- **效果**: +10.7% 提升
- **原理**: 扩展 HCCL 算子的运行时 shape 支持，增加 ACL Graph 捕获机会
- **配置**: `export HCCL_OP_EXPANSION_MODE=AIV`

### 2. 内存配置优化
- **gpu_memory_utilization**: 0.95 (最大化 KV cache 可用内存)
- **max_model_len**: 8192 (平衡并发能力和内存使用)
- **PYTORCH_NPU_ALLOC_CONF**: expandable_segments:True

---

## 目标差距分析

| 指标 | 当前值 | 目标值 (+50%) | 差距 |
|------|--------|--------------|------|
| 输出吞吐量 | 15.91 tok/s | 21.6 tok/s | **+35.8%** |
| 总吞吐量 | 178.83 tok/s | 242.8 tok/s | **+35.8%** |

### 潜在优化空间

| 优化项 | 预期收益 | 实现难度 | 优先级 |
|--------|----------|----------|--------|
| Jemalloc 内存分配器 | +5% | 低 | 高 |
| CPU 亲和性修复 | +3% | 低 | 高 |
| GEMM 融合优化 | +6-8% | 中 | 高 |
| 内存拷贝减少 | +3-4% | 中 | 中 |
| Attention 核优化 | +2-3% | 高 | 中 |
| 更高并发度 | +10-20% | 低 | 高 |

---

## 推荐下一步行动

### 立即可做 (1 天内)
1. 测试并发度 2, 4, 8, 16 的吞吐量
2. 安装 jemalloc 并测试
3. 修复 CPU 亲和性配置

### 短期优化 (1-2 周)
1. 实现 GEMM + Bias 融合核
2. 减少 `copy_`/`clone` 调用
3. 预分配中间缓冲区

### 长期优化 (2-4 周)
1. 自定义 GDN Attention Triton 核
2. 端到端 profiling 驱动优化
3. 探索 KV cache 压缩 (FP8/INT8)

---

## 测试脚本

### 基准测试命令
```bash
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=True
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH

vllm bench throughput --model Qwen/Qwen3.5-0.8B \
  --num-prompts 100 \
  --random-input-len 1024 \
  --random-output-len 100 \
  --dataset-name random \
  --max-num-seqs 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --disable-log-stats
```

### 不同并发度测试
```bash
# 并发度 2
vllm bench throughput --model Qwen/Qwen3.5-0.8B \
  --num-prompts 100 --random-input-len 1024 --random-output-len 100 \
  --dataset-name random --max-num-seqs 2 \
  --max-model-len 8192 --gpu-memory-utilization 0.95 --disable-log-stats

# 并发度 4
vllm bench throughput --model Qwen/Qwen3.5-0.8B \
  --num-prompts 100 --random-input-len 1024 --random-output-len 100 \
  --dataset-name random --max-num-seqs 4 \
  --max-model-len 8192 --gpu-memory-utilization 0.95 --disable-log-stats
```

---

## 关键文件

| 文件 | 描述 |
|------|------|
| `vllm_profile/benchmark_results.md` | 详细基准测试数据 |
| `vllm_profile/performance_optimization_report.md` | 完整优化报告 |
| `vllm_profile/decoding_performance_analysis.md` | 算子级 profiling 分析 |
| `vllm_ascend/attention/attention_v1.py` | Attention 实现 |
| `vllm_ascend/patch/worker/patch_unquantized_gemm.py` | GEMM 补丁 |

---

## 联系与反馈

如需进一步优化或有性能问题反馈，请参考：
- 性能分析指南：`docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`
- 优化与调优：`docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md`
