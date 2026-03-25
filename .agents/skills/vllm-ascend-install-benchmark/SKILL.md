---
name: vllm-ascend-install-benchmark
description: "安装 vLLM Ascend 插件并执行性能基准测试。在 Ascend NPU 上快速部署 vLLM 环境，运行 Qwen3.5-0.8B 模型推理测试，并执行并发吞吐量基准测试（1024 输入/100 输出，并发度 1/20/40）。"
---

# vLLM Ascend 安装与性能测试

## 概述

本技能用于在 Ascend NPU 环境中：
1. 安装和配置 vLLM Ascend 插件（vllm-ascend）
2. 验证模型推理功能（使用 Qwen3.5-0.8B）
3. 执行性能基准测试并生成吞吐量报告

## 前置条件

### 硬件要求
- Atlas 800I A2 推理系列、Atlas A2 训练系列
- Atlas 800I A3 推理系列、Atlas A3 训练系列
- Atlas 300I Duo（实验性）

### 软件要求
- Python >= 3.10, < 3.12
- CANN == 8.5.0
- PyTorch == 2.9.0, torch-npu == 2.9.0

### 环境变量
```bash
# ATB 库路径（必需）
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH

# ModelScope 下载（国内推荐）
export VLLM_USE_MODELSCOPE=True

# NPU 内存配置
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 强制使用 NPU
export VLLM_FORCE_NPU=True
```

## 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
```

### 2. 安装依赖

如果遇到 `arctic-inference` 包冲突，编辑以下文件移除该行：
- `requirements.txt`
- `pyproject.toml`

```bash
# 安装核心依赖
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
pip install torch-npu==2.9.0
pip install triton-ascend==3.2.0
pip install -e .
```

### 3. 验证安装
```bash
python -c "import vllm; print(vllm.__version__)"
```

## 模型推理测试

### 快速测试脚本

创建测试文件 `examples/offline_inference_qwen3_5_0.8b.py`：

```python
import os
os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

from vllm import LLM, SamplingParams

model_id = "Qwen/Qwen3.5-0.8B"
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "人工智能的未来发展",
    "请介绍一下华为公司的历史",
]
sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    top_p=0.8,
    top_k=40,
    repetition_penalty=1.05,
)

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.8,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Input: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text!r}")
```

运行测试：
```bash
python3 examples/offline_inference_qwen3_5_0.8b.py
```

## 性能基准测试

### 测试配置
| 项目 | 配置 |
|------|------|
| **模型** | Qwen/Qwen3.5-0.8B |
| **输入长度** | 1024 tokens (random) |
| **输出长度** | 100 tokens |
| **请求数** | 100 |
| **数据集** | random |
| **并发度** | 1, 20, 40 |

### 离线吞吐量测试（推荐）

在线服务基准测试可能不稳定，推荐使用离线吞吐量测试：

```bash
# 并发度 1
vllm bench throughput --model Qwen/Qwen3.5-0.8B --num-prompts 100 \
  --random-input-len 1024 --random-output-len 100 --dataset-name random \
  --max-num-seqs 1 --disable-log-stats

# 并发度 20
vllm bench throughput --model Qwen/Qwen3.5-0.8B --num-prompts 100 \
  --random-input-len 1024 --random-output-len 100 --dataset-name random \
  --max-num-seqs 20 --disable-log-stats

# 并发度 40（需要降低 max_model_len）
vllm bench throughput --model Qwen/Qwen3.5-0.8B --num-prompts 100 \
  --random-input-len 1024 --random-output-len 100 --dataset-name random \
  --max-num-seqs 40 --max-model-len 131072 --gpu-memory-utilization 0.95 \
  --disable-log-stats
```

### 预期结果参考

| 并发度 | 请求吞吐量 (req/s) | 总令牌吞吐量 (tok/s) | 输出令牌吞吐量 (tok/s) |
|--------|-------------------|---------------------|----------------------|
| 1 | ~0.13 | ~146 | ~16 |
| 20 | ~1.04 | ~1171 | ~104 |
| 40 | - | - | - |

**注意**：并发度 40 可能因 KV cache 内存不足而失败。错误信息：
```
ValueError: To serve at least one request with the models's max seq len (262144),
3.05 GiB KV cache is needed, which is larger than the available KV cache memory (1.48 GiB).
```

解决方案：
- 降低 `--max-model-len` 至 131072 或更低
- 增加 `--gpu-memory-utilization` 至 0.95

### 在线服务测试（可选）

创建 `run_benchmarks.sh`：

```bash
#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_FORCE_NPU=True

MODEL="Qwen/Qwen3.5-0.8B"
SERVER_PORT=8000

# 启动服务器
vllm serve $MODEL \
    --port $SERVER_PORT \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats \
    --disable-log-requests &

SERVER_PID=$!

# 等待服务器就绪
for i in {1..60}; do
    if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# 运行基准测试
vllm bench serve \
    --model $MODEL \
    --base-url http://localhost:$SERVER_PORT \
    --num-prompts 100 \
    --max-concurrency 1 \
    --request-rate inf \
    --random-input-len 1024 \
    --output-len 100 \
    --dataset-name random

# 清理
kill $SERVER_PID 2>/dev/null || true
```

## 常见问题排查

### 1. libatb.so 未找到
```
OSError: libatb.so: cannot open shared object file: No such file or directory
```
**解决**：设置 ATB 库路径
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
```

### 2. Triton 版本不兼容
```
Error: No module named 'triton.language.target_info'
```
**解决**：安装正确的 Triton 版本
```bash
pip install triton-ascend==3.2.0
```

### 3. KV cache 内存不足
```
ValueError: ... 3.05 GiB KV cache is needed, which is larger than the available KV cache memory
```
**解决**：
- 降低 `--max-model-len`（如 131072）
- 增加 `--gpu-memory-utilization`（如 0.95）

### 4. 网络连接问题
```
Max retries exceeded with url: /api/models/... (Caused by NewConnectionError)
```
**解决**：使用 ModelScope 镜像
```bash
export VLLM_USE_MODELSCOPE=True
```

## 性能分析报告模板

测试完成后，整理以下指标：

```markdown
## vLLM Ascend Qwen3.5-0.8B 性能测试报告

### 测试配置
- 硬件：Ascend 910B NPU (CANN 8.5.0)
- 软件：vLLM v0.17.0 + vllm-ascend 插件
- 模型：Qwen/Qwen3.5-0.8B

### 吞吐量结果

| 并发度 | 请求吞吐量 (req/s) | 总令牌吞吐量 (tok/s) | 输出令牌吞吐量 (tok/s) |
|--------|-------------------|---------------------|----------------------|
| 1 | [填写] | [填写] | [填写] |
| 20 | [填写] | [填写] | [填写] |
| 40 | [填写] | [填写] | [填写] |

### 性能分析
- 并发度从 1 提升到 20：吞吐量提升约 8 倍
- KV cache 利用率：[观察结果]
- 编译时间：torch.compile 约 [X] 秒，ACL Graph 捕获约 [Y] 秒

### 问题与解决方案
[记录遇到的问题和解决方法]
```

## 参考文件

- `references/installation.md` — 详细安装指南
- `references/performance_tuning.md` — 性能调优参数
- `references/troubleshooting.md` — 常见问题排查

## 交付物

测试完成后交付：
1. 性能测试报告（包含吞吐量表格）
2. 问题和解决方案记录
3. 可复用的测试脚本（如 `run_benchmarks.sh`）
