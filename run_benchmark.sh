#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_FORCE_NPU=True

echo "=============================================="
echo "vLLM Ascend Qwen3.5-0.8B 基准测试"
echo "=============================================="

# 运行并发度 1 的基准测试
echo "开始并发度 1 测试..."
vllm bench throughput --model Qwen/Qwen3.5-0.8B --num-prompts 100 \
  --random-input-len 1024 --random-output-len 100 --dataset-name random \
  --max-num-seqs 1 --disable-log-stats 2>&1 | tee /tmp/bench_concurrency1.log

echo "基准测试完成！结果保存在 /tmp/bench_concurrency1.log"
