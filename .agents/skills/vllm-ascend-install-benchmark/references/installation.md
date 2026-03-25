# vLLM Ascend 安装指南

## 快速安装

```bash
# 1. 克隆仓库
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# 2. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 3. 安装依赖
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
pip install torch-npu==2.9.0
pip install triton-ascend==3.2.0
pip install -e .
```

## 版本要求

| 组件 | 版本 |
|------|------|
| Python | 3.10 - 3.11 |
| CANN | 8.5.0 |
| PyTorch | 2.9.0 |
| torch-npu | 2.9.0 |
| triton-ascend | 3.2.0 |
| vLLM | 0.17.0 (与 vllm-ascend 同版本) |

## 已知问题

### arctic-inference 冲突

如果安装失败，检查并移除 `arctic-inference==0.1.1`：
- `requirements.txt`
- `pyproject.toml`
