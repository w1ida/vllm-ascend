# Qwen3.5-0.8B NPU Optimization PR Plan

## Scope

This document turns the profiling conclusions for `Qwen/Qwen3.5-0.8B` into a concrete PR-ready work plan for vLLM Ascend.

The plan is intentionally split into small, reviewable changes. Each PR item includes:

- objective
- patch points
- files involved
- expected benefit
- validation focus

## Baseline

Current baseline from the March 21, 2026 profile:

- `366.8 ms` total decode time for 50 output tokens
- `7.34 ms/token`
- major hotspots:
  - `vllm::gdn_attention_core`
  - `vllm::unquantized_gemm`
  - `aten::copy_`
  - `aten::clone`
  - `aten::contiguous`
  - `fused_sigmoid_gating_delta_rule_update_kernel`
  - `_causal_conv1d_update_kernel_npu_tiled`

## PR-01: Add Qwen3.5 Profile Analysis Docs

### Objective

Check in the profiling analysis and optimization roadmap so the work has a stable design reference.

### Patch Points

- add a profiling analysis document
- add a PR implementation plan document
- link both docs into the performance and debug index

### Files

- `docs/source/developer_guide/performance_and_debug/index.md`
- `docs/source/developer_guide/performance_and_debug/qwen3_5_0_8b_npu_decoding_analysis_20260324.md`
- `docs/source/developer_guide/performance_and_debug/qwen3_5_0_8b_npu_optimization_pr_plan_20260324.md`

### Expected Benefit

- no runtime benefit
- reduces implementation ambiguity
- helps architectural review

### Validation

- docs build
- markdown lint / repo formatting checks

## PR-02: Add Qwen3.5 NPU Fused In-Proj Front-End Path

### Objective

Reduce front-end decode overhead for Qwen3.5 by introducing a fused NPU path for projection output split and layout conversion, similar in intent to upstream `gdn_in_proj` and the current Qwen3-Next NPU fused split kernel.

### Patch Points

- add a Qwen3.5-oriented fused helper that directly produces:
  - `mixed_qkv`
  - `z`
  - `b`
  - `a`
- wire Qwen3.5 forward to use the fused path in the non-LoRA fast path
- keep LoRA path behavior unchanged unless separately optimized

### Files

- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py`
- optionally a new file if Qwen3.5 needs a dedicated fused layout kernel:
  - `vllm_ascend/ops/triton/fla/fused_qwen3_5_in_proj.py`
- tests:
  - `tests/ut/patch/worker/`
  - `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/`

### Expected Benefit

- `5%` to `10%` lower end-to-end decode latency
- `20%+` reduction in front-end copy / transpose overhead for the affected path

### Validation

- numerical parity against the current Qwen3.5 path
- no regression for LoRA-disabled inference
- profile comparison for:
  - `copy_`
  - `clone`
  - `contiguous`
  - `Transpose`

## PR-03: Remove Forced Contiguous Materialization in Qwen3.5 Decode Fast Path

### Objective

Reduce repeated `contiguous()`, `clone()`, and copy operations in the decode-only GDN path.

### Patch Points

- audit `rearrange_mixed_qkv` and downstream fast-path inputs
- allow stride-friendly tensors where possible
- pre-layout persistent parameters such as:
  - `A_log`
  - `dt_bias`
- avoid redundant materialization for:
  - `q`
  - `k`
  - `v`
  - `a`
  - `b`

### Files

- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/patch/worker/patch_qwen3_next.py`
- `vllm_ascend/ops/triton/fla/sigmoid_gating.py`
- related custom op wrappers if input layout assumptions need to change

### Expected Benefit

- `10%+` reduction in aggregate:
  - `aten::copy_`
  - `aten::clone`
  - `aten::contiguous`
- `3%` to `8%` lower token latency by itself

### Validation

- correctness tests on decode-only path
- profile diffs before and after
- verify no hidden sync introduced by new layout handling

## PR-04: Build Decode-Only GDN Mega-Fusion Kernel

### Objective

Create a single NPU-oriented fast path for the decode-only GDN chain to reduce launch count and host orchestration overhead.

### Patch Points

- fuse the decode-only chain:
  - conv update
  - QKV split / reshape
  - sigmoid gating delta rule update
  - state update
- add a dispatch branch for decode-only and non-spec decode in Qwen3.5
- preserve existing prefill and spec-decode fallback paths initially

### Files

- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/ops/triton/fla/sigmoid_gating.py`
- `vllm_ascend/ops/triton/fla/fused_qwen3_5_decode.py` or equivalent new kernel file
- potentially `csrc/` bindings if the implementation moves beyond Triton
- tests:
  - `tests/ut/patch/worker/test_patch_qwen3_5*.py`
  - decode-focused e2e or nightly perf tests

### Expected Benefit

- `10%` to `20%` lower `ms/token` on batch-size-1 decode
- large drop in host total duration for `vllm::gdn_attention_core`
- fewer decode-stage small-kernel launches

### Validation

- exact or near-exact output parity
- state update correctness across multiple decode steps
- profile verification:
  - lower `gdn_attention_core` host total duration
  - fewer launches for decode-only GDN path

## PR-05: Enable `float32 ssm_state` in NPU Recurrent GDN Op

### Objective

Remove the current structural blocker that prevents Qwen3.5 from using a more direct recurrent decode path.

### Patch Points

- extend `torch_npu.npu_recurrent_gated_delta_rule` support for `float32 ssm_state`, or add an equivalent supported op path
- replace the temporary fallback logic in Qwen3.5 `_forward_core`
- simplify the control flow once the supported fast path is available

### Files

- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- relevant Ascend op implementation or wrapper layer
- possibly vendor-side or extension-side operator binding files depending on where support must be added

### Expected Benefit

- `5%` to `15%` lower decode latency depending on final kernel quality
- less code divergence between Qwen3.5 and Qwen3-Next fast paths
- reduced maintenance burden

### Validation

- dtype coverage tests
- long decode correctness tests
- no regression in prefill or spec-decode flows

## PR-06: Tighten Hybrid Full-Attention Path for Qwen3.5

### Objective

Reduce the host overhead around the full-attention layers in the hybrid Qwen3.5 stack.

### Patch Points

- audit `unified_attention_with_output`
- audit reshape and cache handling around full-attention layers
- reduce transient tensor creation around KV cache preparation
- improve coupling between cache write and fused attention invocation if feasible

### Files

- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/worker/model_runner_v1.py`
- any cache reshape helper involved in hybrid attention execution

### Expected Benefit

- `2%` to `5%` lower token latency
- lower host total duration for:
  - `vllm::unified_attention_with_output`
  - `npu::npu_fused_infer_attention_score`

### Validation

- hybrid-layer correctness
- cache consistency across long decode
- profile-based before/after comparison

## PR-07: Remove Metadata and Small-Op Work From Decode Hot Path

### Objective

Reduce repeated small-kernel and AI_CPU work in decode mode.

### Patch Points

- precompute or cache:
  - `cu_seqlens`
  - index tensors
  - frequently reused scalar-derived tensors
- remove repeated decode-time creation for values that can be reused
- reduce host-to-device copies for metadata tensors

### Files

- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/patch/worker/patch_qwen3_next.py`
- `vllm_ascend/worker/model_runner_v1.py`
- metadata construction helpers under attention or worker code paths

### Expected Benefit

- `1%` to `3%` lower token latency
- lower counts or cost for:
  - `floor_divide`
  - `cumsum`
  - `argmax`
  - `to`
  - `_to_copy`
  - `acl_memcpy_host_to_device`

### Validation

- regression tests for attention metadata correctness
- profile diff on small-op counts

## PR-08: Add Dedicated Regression and Perf Tests

### Objective

Keep the optimization work stable and measurable.

### Patch Points

- add unit tests for Qwen3.5 fast paths
- add nightly perf regression coverage for:
  - bs1 decode latency
  - copy and transpose overhead
  - GDN path launch count if measurable

### Files

- `tests/ut/patch/worker/`
- `tests/e2e/nightly/single_node/`
- benchmark scripts if needed

### Expected Benefit

- no direct runtime gain
- prevents regression after operator and patch changes

### Validation

- CI pass
- stable perf baseline collection

## Recommended PR Order

Recommended landing order:

1. PR-01 docs
2. PR-02 Qwen3.5 fused in-proj front-end
3. PR-03 contiguous and copy reduction
4. PR-04 decode-only GDN mega-fusion
5. PR-05 `float32 ssm_state` support
6. PR-06 full-attention hybrid cleanup
7. PR-07 metadata hot-path cleanup
8. PR-08 regression and perf coverage

## Stage Exit Targets

### After PR-03

- visible reduction in `copy_`, `clone`, `contiguous`, and `Transpose`
- `5%` to `12%` lower `ms/token`

### After PR-04

- major reduction in GDN decode fragmentation
- `10%` to `20%` lower `ms/token` versus baseline

### After PR-05

- Qwen3.5 decode fast path aligned more closely with intended recurrent update flow
- cumulative `15%` to `25%` latency reduction versus baseline is a reasonable first milestone

## Review Notes

Each PR should clearly state:

- whether it changes only decode path or also prefill path
- whether LoRA path is affected
- whether speculative decode path is affected
- exact profiling scenario used for before and after comparison

This is especially important for Qwen3.5 because the model is hybrid and the hot path spans both GDN and full-attention layers.

