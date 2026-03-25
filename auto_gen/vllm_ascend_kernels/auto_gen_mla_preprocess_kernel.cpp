#ifndef __MLA_PREPROCESS_KERNEL__KERNEL_FUN_H__
#define __MLA_PREPROCESS_KERNEL__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define mla_preprocess mla_preprocess_origin
#include "/workspace/vllm-ascend/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp"

#undef mla_preprocess
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_mla_preprocess_kernel(
GM_ADDR ffts_addr, __attribute__((cce_global)) uint8_t* hiddenState, __attribute__((cce_global)) uint8_t* quantScale1, __attribute__((cce_global)) uint8_t* quantOffset1, __attribute__((cce_global)) uint8_t* wdqkv, __attribute__((cce_global)) uint8_t* bias1, __attribute__((cce_global)) uint8_t* gamma2, __attribute__((cce_global)) uint8_t* beta2, __attribute__((cce_global)) uint8_t* quantScale2, __attribute__((cce_global)) uint8_t* quantOffset2, __attribute__((cce_global)) uint8_t* gamma3, __attribute__((cce_global)) uint8_t* sin1, __attribute__((cce_global)) uint8_t* cos1, __attribute__((cce_global)) uint8_t* sin2, __attribute__((cce_global)) uint8_t* cos2, __attribute__((cce_global)) uint8_t* keycache, __attribute__((cce_global)) uint8_t* slotMapping, __attribute__((cce_global)) uint8_t* wuq, __attribute__((cce_global)) uint8_t* bias2, __attribute__((cce_global)) uint8_t* wuk, __attribute__((cce_global)) uint8_t* descale1, __attribute__((cce_global)) uint8_t* descale2, __attribute__((cce_global)) uint8_t* ctkvScale, __attribute__((cce_global)) uint8_t* qnopeScale, __attribute__((cce_global)) uint8_t* q, __attribute__((cce_global)) uint8_t* keycacheOut, __attribute__((cce_global)) uint8_t* q2, __attribute__((cce_global)) uint8_t* keycacheOut2, __attribute__((cce_global)) uint8_t* innerOut, __attribute__((cce_global)) uint8_t* workspace, __attribute__((cce_global)) uint8_t* tiling, GM_ADDR overflow_status) {
    icache_preload(1);
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));
#endif
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = workspace;
#else
    workspace_param = tiling;
#endif
    if (workspace_param == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(REGIST_MATMUL_OBJ) || defined(__MIX_CORE_MACRO__)
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace_param);
#ifdef ASCENDC_TIME_STAMP_ON
        AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));
#endif
    }
#endif
#if defined(HAVE_TILING)
    workspace = workspace_usr;
#else
    tiling = workspace_usr;
#endif
#endif
    mla_preprocess_origin(hiddenState, quantScale1, quantOffset1, wdqkv, bias1, gamma2, beta2, quantScale2, quantOffset2, gamma3, sin1, cos1, sin2, cos2, keycache, slotMapping, wuq, bias2, wuk, descale1, descale2, ctkvScale, qnopeScale, q, keycacheOut, q2, keycacheOut2, innerOut, workspace, tiling);
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
