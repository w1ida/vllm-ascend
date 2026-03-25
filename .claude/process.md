# vLLM Ascend 快速安装、基准测试与性能优化流程

**适用环境**: Ascend 910B4 NPU, CANN 8.5.0, Python 3.11

---

## 一、环境安装

### 1.1 前置条件检查

```bash
# 检查 NPU 状态
npu-smi info

# 检查 Python 版本
python --version  # 需要 3.10-3.12

# 检查 CANN 环境
cat /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && echo "CANN installed"
```

### 1.2 安装 vLLM 和 vLLM Ascend

```bash
cd /workspace/vllm-ascend

# 1. 移除 arctic-inference 依赖冲突
sed -i '/arctic-inference/d' requirements.txt pyproject.toml

# 2. 安装核心依赖
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
pip install torch-npu==2.9.0
pip install triton-ascend==3.2.0

# 3. 安装 vLLM (使用 no-deps 避免 torch 版本冲突)
pip install vllm==0.17.0 --no-deps

# 4. 安装 vLLM Ascend
pip install -e .

# 5. 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 1.3 修复 API 兼容性问题 (如需要)

如果报错 `get_attn_backend() missing 1 required positional argument: 'block_size'`:

```bash
# 编辑 vllm_ascend/worker/model_runner_v1.py 第 296-303 行
# 添加 block_size=None 参数
```

如果报错 `unexpected keyword argument 'defer_finalize'`:

```bash
# 编辑 vllm_ascend/worker/model_runner_v1.py 第 1381-1386 行
# 将 defer_finalize 改为 clear_metadata
```

---

## 二、基准测试

### 2.1 设置优化环境变量

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
```

### 2.2 运行基准测试

```bash
# 1 并发度测试 (1024 输入 + 100 输出 tokens)
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

### 2.3 多并发度测试

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

# 并发度 8
vllm bench throughput --model Qwen/Qwen3.5-0.8B \
  --num-prompts 100 --random-input-len 1024 --random-output-len 100 \
  --dataset-name random --max-num-seqs 8 \
  --max-model-len 8192 --gpu-memory-utilization 0.95 --disable-log-stats
```

### 2.4 功能验证测试

```bash
python3 << 'EOF'
import os
os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "人工智能的未来发展",
]
sampling_params = SamplingParams(max_tokens=50, temperature=0.7)

llm = LLM(model="Qwen/Qwen3.5-0.8B", max_model_len=2048, gpu_memory_utilization=0.8)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Input: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text!r}")
EOF
```

---

## 三、性能优化

### 3.1 系统级优化

#### 环境变量优化 (已验证 +10.7%)

```bash
# 核心优化变量
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ATB 库路径
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH

# ModelScope 下载加速
export VLLM_USE_MODELSCOPE=True
```

#### Jemalloc 内存优化 (预期 +5%)

```bash
apt-get install -y libjemalloc2
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

#### CPU 亲和性配置

```bash
# 查看 CPU 拓扑
lscpu

# 手动绑定进程到特定 CPU 核
taskset -c 0-15 python3 your_script.py
```

### 3.2 模型配置优化

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_model_len` | 8192 | 平衡 KV cache 和并发能力 |
| `gpu_memory_utilization` | 0.95 | 最大化可用显存 |
| `max-num-seqs` | 1-8 | 根据场景调整 |

### 3.3 Profiling 分析

```bash
# 启用 Ascend Profiler
export ASCEND_PROCESS_LOG=/tmp/vllm_profile
export ASCEND_RT_VISIBLE_DEVICES=0

# 运行 benchmark 生成 profiling 数据
vllm bench throughput --model Qwen/Qwen3.5-0.8B \
  --num-prompts 10 --random-input-len 1024 --random-output-len 100 \
  --dataset-name random --max-num-seqs 1 --disable-log-stats

# 分析 profiling 数据
ls -la /tmp/vllm_profile/rank*/ASCEND_PROFILER_OUTPUT/
```

---

## 四、常见问题排查

### 4.1 libatb.so 未找到

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
```

### 4.2 Triton 版本冲突

```bash
pip uninstall triton -y
pip install triton-ascend==3.2.0
```

### 4.3 TASK_QUEUE_ENABLE 错误

```bash
# 不要设置 TASK_QUEUE_ENABLE=2，与 NPU 图捕获不兼容
unset TASK_QUEUE_ENABLE
```

### 4.4 KV Cache 内存不足

```bash
# 降低 max_model_len 或增加 gpu_memory_utilization
vllm bench throughput ... --max-model-len 4096 --gpu-memory-utilization 0.95
```

### 4.5 HuggingFace 连接失败

```bash
export VLLM_USE_MODELSCOPE=True
pip install modelscope
```

---

## 五、性能基准参考

### 5.1 1 并发测试结果 (Qwen3.5-0.8B)

| 配置 | 输出吞吐量 (tok/s) | 提升 |
|------|-------------------|------|
| 基线 | 14.40 | - |
| + HCCL_OP_EXPANSION_MODE=AIV | 15.94 | +10.7% |
| + max_model_len=8192 | 15.47 | +7.4% |
| 最终配置 | 15.91 | +10.5% |

### 5.2 主要瓶颈算子

| 算子 | 耗时 (ms) | 占比 | 优化方向 |
|------|-----------|------|----------|
| gdn_attention_core | 48.80 | 13.3% | 自定义 Triton 核 |
| unquantized_gemm | 36.17 | 9.9% | CANN MMLH 融合 |
| copy_/clone | 36.62 | 10.0% | 预分配缓冲区 |

---

## 六、一键测试脚本

创建 `quick_bench.sh`:

```bash
#!/bin/bash
set -e

# 环境配置
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV

echo "=== vLLM Ascend Benchmark ==="
echo "Model: Qwen/Qwen3.5-0.8B"
echo "Concurrency: $1 (default: 1)"

MAX_NUM_SEQS=${1:-1}

vllm bench throughput --model Qwen/Qwen3.5-0.8B \
  --num-prompts 100 \
  --random-input-len 1024 \
  --random-output-len 100 \
  --dataset-name random \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --disable-log-stats

echo "=== Benchmark Complete ==="
```

使用:
```bash
chmod +x quick_bench.sh
./quick_bench.sh 1   # 1 并发
./quick_bench.sh 4   # 4 并发
```

---

## 七、参考文件

| 文件 | 描述 |
|------|------|
| `vllm_profile/benchmark_results.md` | 详细基准测试数据 |
| `vllm_profile/performance_optimization_report.md` | 完整优化报告 |
| `vllm_profile/OPTIMIZATION_SUMMARY.md` | 优化总结 |
| `docs/source/installation.md` | 安装指南 |
| `docs/source/developer_guide/performance_and_debug/` | 性能调试文档 |

---

## 八、快速参考命令

```bash
# 检查 NPU 状态
npu-smi info

# 验证 vLLM 安装
python -c "import vllm; print(vllm.__version__)"

# 运行基准测试
vllm bench throughput --model Qwen/Qwen3.5-0.8B --num-prompts 100 --random-input-len 1024 --random-output-len 100 --dataset-name random --max-num-seqs 1 --disable-log-stats

# 清理缓存
rm -rf /root/.cache/vllm/torch_compile_cache/*
```

---

**最后更新**: 2026-03-22
**测试环境**: Ascend 910B4, CANN 8.5.0, vLLM 0.17.0, vllm-ascend main
