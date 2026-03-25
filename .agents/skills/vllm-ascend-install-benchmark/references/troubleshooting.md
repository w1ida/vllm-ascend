# 常见问题排查

## 安装问题

### 1. libatb.so 未找到
```
OSError: libatb.so: cannot open shared object file: No such file or directory
```

**原因**：ATB 库路径未设置

**解决**：
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
```

### 2. torch 版本冲突
```
vllm 0.17.0 installed torch 2.10.0 but torch-npu 2.9.0 requires torch 2.9.0
```

**原因**：vLLM 自动安装了不兼容的 torch 版本

**解决**：
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

### 3. triton 导入失败
```
Error: No module named 'triton.language.target_info'
```

**原因**：Triton 版本不兼容

**解决**：
```bash
pip install triton-ascend==3.2.0
```

### 4. arctic-inference 安装失败
```
ERROR: Could not find a version that satisfies the requirement arctic-inference==0.1.1
```

**原因**：包不可用或版本冲突

**解决**：编辑以下文件，移除 `arctic-inference==0.1.1` 行：
- `requirements.txt`
- `pyproject.toml`

## 运行时问题

### 5. KV cache 内存不足
```
ValueError: To serve at least one request with the models's max seq len (262144),
3.05 GiB KV cache is needed, which is larger than the available KV cache memory (1.48 GiB).
```

**原因**：max_model_len 过大导致 KV cache 需求超出可用内存

**解决**：
```bash
# 方案 1：降低最大模型长度
--max-model-len 131072

# 方案 2：增加 GPU 内存利用率
--gpu-memory-utilization 0.95
```

### 6. 模型下载失败
```
Max retries exceeded with url: /api/models/... (Caused by NewConnectionError)
```

**原因**：HuggingFace 连接超时

**解决**：
```bash
export VLLM_USE_MODELSCOPE=True
```

### 7. EngineDeadError
```
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue
```

**原因**：服务器在高负载下崩溃

**解决**：
- 使用离线基准测试（`vllm bench throughput`）代替在线测试
- 降低并发度
- 减少 `max-num-seqs`

### 8. ACL Graph 编译失败
```
ERROR: ACL graph compilation failed
```

**原因**：编译配置问题

**解决**：
```bash
# 检查编译日志
cat /tmp/vllm_server.log | grep -i "acl\|compile"

# 尝试降低并发批大小
--max-num-batched-tokens 4096
```

## 性能问题

### 9. 吞吐量低于预期

**可能原因**：
- GPU 内存利用率过低
- KV cache 不足导致频繁换入换出
- 编译未生效

**排查步骤**：
1. 检查 `gpu-memory-utilization` 是否 >= 0.9
2. 检查日志中的 KV cache 大小
3. 确认 ACL Graph 编译成功（日志中应有 "Graph capturing finished"）

### 10. 延迟过高

**可能原因**：
- 并发度过高导致资源争用
- 批处理过大
- CPU 绑定不当

**解决**：
- 降低并发度
- 调整 `max-num-batched-tokens`
- 检查 CPU 绑定日志

## 日志位置

| 日志类型 | 位置 |
|---------|------|
| 服务器日志 | `/tmp/vllm_server.log` |
| 基准测试日志 | 当前终端输出 |
| 编译日志 | 标准输出（带 [compilation] 标签） |
| NPU 日志 | `var/log/ascend/` |

## 获取帮助

- 文档：https://docs.vllm.ai/projects/ascend/
- 问题反馈：https://github.com/vllm-project/vllm-ascend/issues
- 用户论坛：https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support
