# 性能调优参数

## vLLM 启动参数

### 内存管理
| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--gpu-memory-utilization` | 0.9 | 0.9 - 0.95 | GPU 内存利用率，高并发时可调至 0.95 |
| `--max-model-len` | 自动 | 131072 | 最大模型长度，高并发时降低以节省 KV cache |

### 并发控制
| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--max-num-seqs` | 256 | 根据并发调整 | 最大序列数 |
| `--max-num-batched-tokens` | 8192 | 8192 | 批处理 token 数 |

### 编译优化
| 参数 | 说明 |
|------|------|
| `--compilation-config` | ACL Graph 编译配置 |
| `--enable-chunked-prefill` | 启用分块预填充 |

## 基准测试参数

### vllm bench throughput
```bash
vllm bench throughput \
  --model <模型路径> \
  --num-prompts 100 \
  --random-input-len 1024 \
  --random-output-len 100 \
  --dataset-name random \
  --max-num-seqs <并发度> \
  --disable-log-stats
```

### vllm bench serve
```bash
vllm bench serve \
  --model <模型路径> \
  --base-url http://localhost:8000 \
  --num-prompts 100 \
  --max-concurrency <并发度> \
  --request-rate inf \
  --random-input-len 1024 \
  --output-len 100 \
  --dataset-name random
```

## 性能指标解读

### 吞吐量指标
- **requests/s**: 每秒处理的请求数
- **total tokens/s**: 每秒处理的总 token 数（输入 + 输出）
- **output tokens/s**: 每秒生成的输出 token 数

### 延迟指标
- **Avg latency**: 平均延迟
- **P50/P90/P99 latency**: 延迟百分位数

## 优化建议

1. **高并发场景**：降低 `max-model-len`，增加 `gpu-memory-utilization`
2. **长文本场景**：启用 chunked prefill，增加 `max-num-batched-tokens`
3. **KV cache 不足**：减少并发度或降低 `max-model-len`
