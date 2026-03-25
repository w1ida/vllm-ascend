#ifndef __BATCH_MATMUL_TRANSPOSE_KERNEL__KERNEL_FUN_H__
#define __BATCH_MATMUL_TRANSPOSE_KERNEL__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define batch_matmul_transpose batch_matmul_transpose_origin
#include "/workspace/vllm-ascend/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp"

#undef batch_matmul_transpose
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_batch_matmul_transpose_kernel(
__attribute__((cce_global)) uint8_t* gm_a, __attribute__((cce_global)) uint8_t* gm_b, __attribute__((cce_global)) uint8_t* gm_c, __attribute__((cce_global)) uint8_t* gm_tiling_data, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = gm_c;
#else
    workspace_param = gm_tiling_data;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    gm_c = workspace_usr;
#else
    gm_tiling_data = workspace_usr;
#endif
#endif
    batch_matmul_transpose_origin(gm_a, gm_b, gm_c, gm_tiling_data);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

static const struct FunLevelKType batch_matmul_transpose_section __attribute__ ((used, section (".ascend.meta.batch_matmul_transpose_0"))) = { { {F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_AIC} };
#endif
