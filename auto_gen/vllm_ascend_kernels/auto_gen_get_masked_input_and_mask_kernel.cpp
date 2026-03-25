#ifndef __GET_MASKED_INPUT_AND_MASK_KERNEL__KERNEL_FUN_H__
#define __GET_MASKED_INPUT_AND_MASK_KERNEL__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define get_masked_input_and_mask_kernel get_masked_input_and_mask_kernel_origin
#include "/workspace/vllm-ascend/csrc/kernels/get_masked_input_and_mask_kernel.cpp"

#undef get_masked_input_and_mask_kernel
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_get_masked_input_and_mask_kernel_kernel(
__attribute__((cce_global)) int32_t* input, __attribute__((cce_global)) int32_t* masked_input, __attribute__((cce_global)) bool* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = loop_cnt;
#else
    workspace_param = aiv_num;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    loop_cnt = workspace_usr;
#else
    aiv_num = workspace_usr;
#endif
#endif
    get_masked_input_and_mask_kernel_origin(input, masked_input, mask_out, org_vocab_start_index, org_vocab_end_index, num_org_vocab_padding, added_vocab_start_index, added_vocab_end_index, size, loop_cnt, aiv_num);
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
