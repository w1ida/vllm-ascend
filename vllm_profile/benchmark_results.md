# vLLM Ascend Performance Benchmark Results

## Test Configuration
- **Model**: Qwen/Qwen3.5-0.8B
- **Hardware**: Ascend 910B4 NPU (CANN 8.5.0)
- **Software**: vLLM v0.17.0 + vllm-ascend main branch
- **Input Length**: 1024 tokens (random)
- **Output Length**: 100 tokens
- **Prompts**: 100
- **Concurrency**: max-num-seqs=1

## Benchmark Results Summary

### System-Level Optimizations

| Configuration | Output tokens/s | Total tokens/s | Improvement | Time (mm:ss) |
|--------------|-----------------|----------------|-------------|--------------|
| **Baseline** (no optimizations) | 14.40 | 161.85 | - | 11:33 |
| + HCCL_OP_EXPANSION_MODE=AIV | 15.94 | 179.21 | +10.7% | 10:27 |
| + max_model_len=8192 | 15.47 | 173.89 | +7.4% | 10:45 |
| **Final** (AIV + max_model_len=8192 + gpu_mem=0.95) | 15.91 | 178.83 | +10.5% | 10:27 |

### Environment Variables Used
```bash
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=True
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
```

### Failed Optimization Attempts
- `TASK_QUEUE_ENABLE=2`: Incompatible with NPU graph capture
  - Error: "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture"

## Key Findings

1. **HCCL_OP_EXPANSION_MODE=AIV** provides ~10% improvement by enabling more runtime shapes support
2. **Reduced max_model_len** (4096 -> 8192) allows more efficient memory utilization
3. **Higher gpu_memory_utilization** (0.95) provides more KV cache space

## Performance Gap to Target

- **Current**: 15.91 output tokens/s
- **Target** (50% improvement): 21.6 output tokens/s
- **Gap**: 5.69 tokens/s (+35.8% more improvement needed)

## Next Optimization Opportunities

### 1. Operator-Level Optimization (requires profiling)
- Target top bottlenecks from existing profiling data:
  - `vllm::gdn_attention_core` (48.80ms, 900 calls)
  - `vllm::unquantized_gemm` (36.17ms, 164 calls)
  - `aten::copy_` (19.73ms, 4211 calls)

### 2. Higher Concurrency Testing
- Existing data shows concurrency=20 achieved ~104 output tokens/s
- Test concurrency=2, 4, 8 for single-request optimization

### 3. Additional System Optimizations
- Jemalloc memory allocator (requires root)
- CPU affinity tuning
- Python with LTO/PGO optimizations

## Profiling Data Location
- `/tmp/vllm_profile/rank0_341923_20260321145959955_ascend_pt/`
- `ASCEND_PROFILER_OUTPUT/operator_details.csv`
- `ASCEND_PROFILER_OUTPUT/trace_view.json`
