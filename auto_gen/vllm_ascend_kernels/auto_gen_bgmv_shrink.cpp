#ifndef __BGMV_SHRINK__KERNEL_FUN_H__
#define __BGMV_SHRINK__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define bgmv_shrink_bfloat16_t bgmv_shrink_bfloat16_t_origin
#define bgmv_shrink_half bgmv_shrink_half_origin
#include "/workspace/vllm-ascend/csrc/kernels/bgmv_shrink.cpp"

#undef bgmv_shrink_bfloat16_t
#undef bgmv_shrink_half
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_bgmv_shrink_bfloat16_t_kernel(
__attribute__((cce_global)) void* x, __attribute__((cce_global)) void* weight, __attribute__((cce_global)) void* indices, uint32_t indicesSize, __attribute__((cce_global)) void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = maxLoRARank;
#else
    workspace_param = scale;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    maxLoRARank = workspace_usr;
#else
    scale = workspace_usr;
#endif
#endif
    bgmv_shrink_bfloat16_t_origin(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

extern "C" __global__ [aicore] void auto_gen_bgmv_shrink_half_kernel(
__attribute__((cce_global)) void* x, __attribute__((cce_global)) void* weight, __attribute__((cce_global)) void* indices, uint32_t indicesSize, __attribute__((cce_global)) void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = maxLoRARank;
#else
    workspace_param = scale;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    maxLoRARank = workspace_usr;
#else
    scale = workspace_usr;
#endif
#endif
    bgmv_shrink_half_origin(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
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
