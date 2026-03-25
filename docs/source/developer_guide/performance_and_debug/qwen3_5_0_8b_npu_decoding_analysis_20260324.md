# Qwen3.5-0.8B Decode Profiling Analysis on NPU

## Overview

This document summarizes the profiling analysis for `Qwen/Qwen3.5-0.8B` on vLLM Ascend using the profile package uploaded under `0324/`.

Analysis date: March 25, 2026

Profile artifacts used in this analysis:

- `0324/vllm_profile/decoding_performance_analysis.md`
- `0324/vllm_profile/rank0_341923_20260321145959955_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv`
- `0324/vllm_profile/rank0_341923_20260321145959955_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv`
- `0324/vllm_profile/rank0_341923_20260321145959955_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`

The goal is to compare the current NPU execution characteristics with the upstream CUDA-oriented vLLM implementation path and derive operator-level optimization directions for NPU.

## Test Configuration

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-0.8B` |
| Runtime | vLLM Ascend |
| `max-num-seqs` | 1 |
| `max_model_len` | 4096 |
| Input tokens | 5 |
| Output tokens | 50 |
| Profiling dataset time | March 21, 2026 |

## Key Metrics

| Metric | Value |
| --- | --- |
| Total decode time | `366.8 ms` |
| Average token latency | `7.34 ms/token` |
| Throughput | `~136 tokens/s` |

This is a batch-size-1 decode profile. In this regime, launch overhead, layout conversion, metadata preparation, and state update overhead are amplified.

## High-Level Bottleneck Breakdown

The profiler summary shows the following category split:

| Category | Time | Ratio |
| --- | ---: | ---: |
| GEMM / MatMul | `144.93 ms` | `39.51%` |
| Other | `101.39 ms` | `27.64%` |
| Attention | `67.85 ms` | `18.50%` |
| Memory Copy | `39.69 ms` | `10.82%` |
| GDN / Delta Rule | `12.33 ms` | `3.36%` |

At first glance, GEMM is the largest bucket. However, a closer inspection shows that the decode path is not bottlenecked only by a single large compute kernel. The more important pattern is that the GDN decode path is fragmented into many small kernels and layout conversion steps.

## Top Operators

Top operators reported in the profile package:

| Rank | Operator | Count | Device Time |
| --- | --- | ---: | ---: |
| 1 | `vllm::gdn_attention_core` | 900 | `48.80 ms` |
| 2 | `vllm::unquantized_gemm` | 164 | `36.17 ms` |
| 3 | `aten::copy_` | 4211 | `19.73 ms` |
| 4 | `aclnnInplaceCopy` | 2799 | `17.68 ms` |
| 5 | `aten::clone` | 1176 | `16.89 ms` |
| 6 | `aten::contiguous` | 1050 | `16.68 ms` |
| 7 | `ChunkGatedDeltaRuleFunction` | 18 | `12.33 ms` |
| 8 | `fused_sigmoid_gating_delta_rule_update_kernel` | 882 | `11.18 ms` |
| 9 | `vllm::unified_attention_with_output` | 300 | `6.98 ms` |
| 10 | `_causal_conv1d_update_kernel` | 882 | `6.29 ms` |

## Device-Side Kernel Hotspots

Aggregated by `op_statistic.csv`, the most expensive kernels on device are:

| Kernel Type | Core | Count | Device Time | Ratio |
| --- | --- | ---: | ---: | ---: |
| `MatMulV2` | `AI_CORE` | 5456 | `129489.9 us` | `55.51%` |
| `Transpose` | `AI_VECTOR_CORE` | 1008 | `17029.52 us` | `7.30%` |
| `fused_sigmoid_gating_delta_rule_update_kernel` | `AI_VECTOR_CORE` | 882 | `11179.24 us` | `4.79%` |
| `_causal_conv1d_update_kernel_npu_tiled` | `AI_VECTOR_CORE` | 882 | `6293.20 us` | `2.70%` |
| `_layer_norm_fwd_1pass_kernel_npu_0` | `AI_VECTOR_CORE` | 882 | `6266.42 us` | `2.69%` |
| `FusedInferAttentionScore` | `MIX_AIC` | 300 | `6024.70 us` | `2.58%` |
| `AddRmsNormBias` | `AI_VECTOR_CORE` | 2400 | `5226.56 us` | `2.24%` |
| `MatMulCommon` | `AI_CORE` | 294 | `5091.28 us` | `2.18%` |
| `FloorDiv` | `AI_CPU` | 126 | `5086.62 us` | `2.18%` |
| `SwiGlu` | `AI_VECTOR_CORE` | 1200 | `4562.26 us` | `1.96%` |

## Host-Side Overhead Is Also Significant

`operator_details.csv` shows a key pattern: several operators have much larger host total duration than device duration.

Representative examples:

| Operator | Count | Host Total | Device Total |
| --- | ---: | ---: | ---: |
| `vllm::gdn_attention_core` | 900 | `2459714.20 us` | `48795.14 us` |
| `ChunkGatedDeltaRuleFunction` | 18 | `419401.58 us` | `12329.76 us` |
| `vllm::unified_attention_with_output` | 300 | `349991.69 us` | `6982.30 us` |
| `aten::copy_` | 4211 | `344550.04 us` | `19726.18 us` |
| `npu::npu_fused_infer_attention_score` | 300 | `138125.52 us` | `6036.02 us` |
| `npu_fx_compiler inference` | 25 | `112168.67 us` | `2966.44 us` |

This suggests that the current decode path has substantial overhead from:

- operator dispatch and orchestration
- repeated shape/layout handling
- temporary tensor allocation and materialization
- synchronization-sensitive state transitions
- fragmented kernel launch patterns

For batch-size-1 decode, this overhead is often as important as raw kernel compute time.

## Memory and Layout Overhead

The profile shows strong evidence of layout conversion and tensor materialization overhead:

- `aten::copy_`: 4211 calls
- `aclnnInplaceCopy`: 2799 calls
- `aten::clone`: 1176 calls
- `aten::contiguous`: 1050 calls
- `Transpose`: 1008 kernel launches

This indicates that the decode path is paying a large cost to rearrange tensors into kernel-friendly layouts. On NPU, these repeated transforms are especially harmful in a small-batch decode workload.

## CPU and Metadata Work

The profile also contains small but non-trivial metadata operators:

| Operator | Count | Host Total | Device Total |
| --- | ---: | ---: | ---: |
| `aten::floor_divide` | 147 | `51353.75 us` | `5435.34 us` |
| `aten::cumsum` | 144 | `18447.17 us` | `1685.92 us` |
| `aten::argmax` | 101 | `15391.34 us` | `3333.08 us` |
| `acl_memcpy_host_to_device` | 707 | `16564.65 us` | `1938.44 us` |
| `aten::to` | 2678 | `96460.50 us` | `1006.44 us` |
| `aten::_to_copy` | 653 | `79351.10 us` | `241.76 us` |

These are not the primary device bottlenecks, but in decode mode they are part of the latency tax and should be reduced where possible.

## Comparison With the CUDA-Oriented Upstream Path

The upstream vLLM implementation for Qwen3.5 and Qwen3-Next already points to the intended high-level design:

- Qwen3.5 forward uses `torch.ops.vllm.gdn_in_proj` to reduce front-end projection and split overhead.
- Qwen3-Next conditionally uses more fused GDN backends on CUDA, including packed recurrent decode and FlashInfer-based paths where applicable.

Reference upstream source files:

- `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_5.py`
- `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next.py`

Compared with that direction, the current NPU path still shows three gaps:

1. Qwen3.5 NPU decode is not fused enough at the front of the GDN pipeline.
2. Layout conversion remains expensive.
3. The recurrent decode path cannot fully use the most direct NPU fast path because `float32 ssm_state` is not yet supported by the needed op.

## Current NPU Implementation Observations

Relevant local implementation points:

- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/patch/worker/patch_qwen3_next.py`
- `vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py`

Important observations:

### Qwen3.5 still carries a temporary fallback path

In `patch_qwen3_5.py`, the comment states that the current implementation is retained because `torch_npu.npu_recurrent_gated_delta_rule` does not support `float32` `ssm_state` inputs yet. This is a structural blocker for a more direct fused recurrent decode path.

### Qwen3-Next already has a fused split/rearrange kernel

`patch_qwen3_next.py` uses `fused_qkvzba_split_reshape_cat`, which shows the right direction for reducing front-end layout work on NPU.

### Qwen3.5 still pays repeated `contiguous()` costs

The Qwen3.5 decode path repeatedly materializes contiguous tensors for `A_log`, `dt_bias`, `q`, `k`, `v`, `a`, and `b`, which is consistent with the copy and clone hotspots seen in the profile.

## Root Cause Summary

The main issue is not simply "MatMul is slow". The deeper problem is:

The GDN decode path is over-fragmented for NPU batch-size-1 serving.

That fragmentation appears in the form of:

- separate conv update, split, reshape, gating, recurrent update, norm, and output projection stages
- repeated tensor layout conversions
- repeated temporary tensor creation
- expensive host-side orchestration relative to actual device work

## Operator-Level Optimization Directions

### Priority 0: Build a decode-only GDN mega-fusion path

Fuse as much of the following chain as possible into a single NPU-oriented decode fast path:

- `causal_conv1d_update`
- mixed QKV split and reshape
- `sigmoid_gating_delta_rule_update`
- state read and write
- optional output layout write-back

Expected effect:

- reduce launch count
- reduce tensor materialization
- reduce `copy_`, `clone`, `contiguous`, and `Transpose`
- reduce host orchestration cost in `gdn_attention_core`

### Priority 0: Add a Qwen3.5-specific NPU fused in-proj and layout path

Qwen3.5 should gain an NPU path analogous in intent to upstream `gdn_in_proj` and to the existing Qwen3-Next NPU fused split kernel.

Expected effect:

- reduce front-end rearrange overhead
- reduce device-side transpose and copy overhead
- reduce host-side tensor preparation cost

### Priority 0: Make hot kernels accept stride-friendly inputs

Instead of forcing `.contiguous()` before every decode update, prefer kernels that accept the layout produced by the previous stage or pre-arrange persistent parameters once.

Expected effect:

- direct reduction in `aten::contiguous`, `aten::clone`, and `aten::copy_`
- lower temporary allocation pressure

### Priority 1: Enable `float32 ssm_state` support in the recurrent GDN op

This is required to remove the current Qwen3.5 fallback path and align the implementation more closely with the intended fast decode route.

Expected effect:

- shorter recurrent path
- less fallback logic in `_forward_core`
- lower host overhead and less state conversion work

### Priority 1: Tighten full-attention cache and attention coupling

The full-attention path is not the top bottleneck, but it still has non-trivial host overhead around cache reshape and fused attention calls.

Expected effect:

- modest latency reduction
- cleaner behavior for hybrid layers in Qwen3.5

### Priority 2: Remove repeated metadata work from the decode hot path

Precompute or cache values used for:

- `floor_divide`
- `cumsum`
- `argmax`
- `to` / `_to_copy`

Expected effect:

- incremental token latency reduction
- less AI_CPU and small-kernel noise

## What Should Not Be Prioritized First

The following items may still be useful at service level, but they should not be treated as the main operator-level solution for this profile:

- increasing concurrency
- reducing `max_model_len`
- switching allocators
- Python runtime rebuild
- generic environment tuning only

These are secondary compared with decode-path fusion and layout reduction.

## Recommended Optimization Order

1. Decode-only GDN fusion for Qwen3.5
2. Qwen3.5 fused in-proj and split/rearrange path
3. Removal of forced contiguous and clone-heavy transitions
4. `float32 ssm_state` support for recurrent GDN update
5. Hybrid full-attention cache and attention path cleanup
6. Metadata and small-op hot-path cleanup

## Success Criteria

The optimization work should target both device efficiency and end-to-end token latency.

Primary success metrics:

- lower `ms/token` in batch-size-1 decode
- fewer launches in the GDN decode chain
- lower total time in `copy_`, `clone`, `contiguous`, and `Transpose`
- lower host total duration for `vllm::gdn_attention_core`

Recommended acceptance threshold for the first optimization round:

- `15%` to `25%` reduction in single-token decode latency for `Qwen/Qwen3.5-0.8B`
- `30%+` reduction in aggregate copy and contiguous overhead in the same profile scenario

