#ifndef __SGMV_EXPAND__KERNEL_FUN_H__
#define __SGMV_EXPAND__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define sgmv_expand_bfloat16_t sgmv_expand_bfloat16_t_origin
#define sgmv_expand_half sgmv_expand_half_origin
#include "/workspace/vllm-ascend/csrc/kernels/sgmv_expand.cpp"

#undef sgmv_expand_bfloat16_t
#undef sgmv_expand_half
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_sgmv_expand_bfloat16_t_kernel(
__attribute__((cce_global)) void* x, __attribute__((cce_global)) void* weight, __attribute__((cce_global)) void* loraIndices, uint32_t loraIndicesSize, __attribute__((cce_global)) void* seqLen, uint32_t seqLenSize, __attribute__((cce_global)) void* yIn, __attribute__((cce_global)) void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = sliceOffset;
#else
    workspace_param = outputFullDim;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    sliceOffset = workspace_usr;
#else
    outputFullDim = workspace_usr;
#endif
#endif
    sgmv_expand_bfloat16_t_origin(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

extern "C" __global__ [aicore] void auto_gen_sgmv_expand_half_kernel(
__attribute__((cce_global)) void* x, __attribute__((cce_global)) void* weight, __attribute__((cce_global)) void* loraIndices, uint32_t loraIndicesSize, __attribute__((cce_global)) void* seqLen, uint32_t seqLenSize, __attribute__((cce_global)) void* yIn, __attribute__((cce_global)) void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = sliceOffset;
#else
    workspace_param = outputFullDim;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    sliceOffset = workspace_usr;
#else
    outputFullDim = workspace_usr;
#endif
#endif
    sgmv_expand_half_origin(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

#endif
