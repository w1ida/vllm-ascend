#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

static char ascendcErrMsg[1024] = {0};

static void *g_kernel_handle = nullptr;

static void *g_kernel_handle_aiv = nullptr;

static void *g_kernel_handle_aic = nullptr;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t mix_type;
    uint32_t mix_len;
    uint32_t mix_file_len;
    uint8_t mix_buf[182248];
    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[61168];
    uint32_t aic_type;
    uint32_t aic_len;
    uint32_t aic_file_len;
    uint8_t aic_buf[72152];
} __ascend_kernel_ascend910b4_vllm_ascend_kernels __attribute__ ((section (".ascend.kernel.ascend910b4.vllm_ascend_kernels"))) = {1,3,0,182248,182248,{0},1,61168,61168,{0},2,72152,72152,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
}

namespace Adx {
    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
            if (g_kernel_handle) {
                UnregisterAscendBinary(g_kernel_handle);
                g_kernel_handle = nullptr;
            }
            if (g_kernel_handle_aiv) {
                UnregisterAscendBinary(g_kernel_handle_aiv);
                g_kernel_handle_aiv = nullptr;
            }
            if (g_kernel_handle_aic) {
                UnregisterAscendBinary(g_kernel_handle_aic);
                g_kernel_handle_aic = nullptr;
            }
        }
    };

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "ascend910b4";
    uint32_t ret;

    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b4_vllm_ascend_kernels.mix_buf,
        __ascend_kernel_ascend910b4_vllm_ascend_kernels.mix_file_len,
        0,
        &g_kernel_handle);
    if (ret != 0) {
        printf("RegisterAscendBinary mix ret %u \n", ret);
    }

    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b4_vllm_ascend_kernels.aiv_buf,
        __ascend_kernel_ascend910b4_vllm_ascend_kernels.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b4_vllm_ascend_kernels.aic_buf,
        __ascend_kernel_ascend910b4_vllm_ascend_kernels.aic_file_len,
        2,
        &g_kernel_handle_aic);
    if (ret != 0) {
        printf("RegisterAscendBinary aic ret %u \n", ret);
    }

    AscendProfRegister();
}





uint32_t launch_and_profiling_bgmv_expand_bfloat16_t(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "bgmv_expand_bfloat16_t";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_bgmv_expand_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* indices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t indicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yIn;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yOut;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t sliceOffset;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputFullDim;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.indices = indices;
    __ascendc_args.indicesSize = indicesSize;
    __ascendc_args.yIn = yIn;
    __ascendc_args.yOut = yOut;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.outputHiddenDim = outputHiddenDim;
    __ascendc_args.sliceOffset = sliceOffset;
    __ascendc_args.outputFullDim = outputFullDim;

    __ascendc_ret = launch_and_profiling_bgmv_expand_bfloat16_t(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_bgmv_expand_half(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "bgmv_expand_half";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_bgmv_expand_half(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* indices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t indicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yIn;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yOut;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t sliceOffset;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputFullDim;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.indices = indices;
    __ascendc_args.indicesSize = indicesSize;
    __ascendc_args.yIn = yIn;
    __ascendc_args.yOut = yOut;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.outputHiddenDim = outputHiddenDim;
    __ascendc_args.sliceOffset = sliceOffset;
    __ascendc_args.outputFullDim = outputFullDim;

    __ascendc_ret = launch_and_profiling_bgmv_expand_half(1, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_bgmv_shrink_bfloat16_t(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "bgmv_shrink_bfloat16_t";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_bgmv_shrink_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* indices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t indicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t inputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(float) + 3) >> 2) << 2) float scale;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.indices = indices;
    __ascendc_args.indicesSize = indicesSize;
    __ascendc_args.y = y;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.inputHiddenDim = inputHiddenDim;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.scale = scale;

    __ascendc_ret = launch_and_profiling_bgmv_shrink_bfloat16_t(2, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_bgmv_shrink_half(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "bgmv_shrink_half";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_bgmv_shrink_half(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* indices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t indicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t inputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(float) + 3) >> 2) << 2) float scale;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.indices = indices;
    __ascendc_args.indicesSize = indicesSize;
    __ascendc_args.y = y;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.inputHiddenDim = inputHiddenDim;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.scale = scale;

    __ascendc_ret = launch_and_profiling_bgmv_shrink_half(3, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_get_masked_input_and_mask_kernel(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "get_masked_input_and_mask_kernel";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_get_masked_input_and_mask_kernel(uint32_t blockDim, void* stream, void* input, void* masked_input, void* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* input;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* masked_input;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* mask_out;
        alignas(((alignof(int64_t) + 3) >> 2) << 2) int64_t org_vocab_start_index;
        alignas(((alignof(int64_t) + 3) >> 2) << 2) int64_t org_vocab_end_index;
        alignas(((alignof(int64_t) + 3) >> 2) << 2) int64_t num_org_vocab_padding;
        alignas(((alignof(int64_t) + 3) >> 2) << 2) int64_t added_vocab_start_index;
        alignas(((alignof(int64_t) + 3) >> 2) << 2) int64_t added_vocab_end_index;
        alignas(((alignof(int64_t) + 3) >> 2) << 2) int64_t size;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t loop_cnt;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t aiv_num;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.input = input;
    __ascendc_args.masked_input = masked_input;
    __ascendc_args.mask_out = mask_out;
    __ascendc_args.org_vocab_start_index = org_vocab_start_index;
    __ascendc_args.org_vocab_end_index = org_vocab_end_index;
    __ascendc_args.num_org_vocab_padding = num_org_vocab_padding;
    __ascendc_args.added_vocab_start_index = added_vocab_start_index;
    __ascendc_args.added_vocab_end_index = added_vocab_end_index;
    __ascendc_args.size = size;
    __ascendc_args.loop_cnt = loop_cnt;
    __ascendc_args.aiv_num = aiv_num;

    __ascendc_ret = launch_and_profiling_get_masked_input_and_mask_kernel(4, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_sgmv_expand_bfloat16_t(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "sgmv_expand_bfloat16_t";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_sgmv_expand_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* loraIndices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t loraIndicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* seqLen;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t seqLenSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yIn;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yOut;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t sliceOffset;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputFullDim;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.loraIndices = loraIndices;
    __ascendc_args.loraIndicesSize = loraIndicesSize;
    __ascendc_args.seqLen = seqLen;
    __ascendc_args.seqLenSize = seqLenSize;
    __ascendc_args.yIn = yIn;
    __ascendc_args.yOut = yOut;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.outputHiddenDim = outputHiddenDim;
    __ascendc_args.sliceOffset = sliceOffset;
    __ascendc_args.outputFullDim = outputFullDim;

    __ascendc_ret = launch_and_profiling_sgmv_expand_bfloat16_t(5, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_sgmv_expand_half(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "sgmv_expand_half";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_sgmv_expand_half(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* loraIndices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t loraIndicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* seqLen;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t seqLenSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yIn;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* yOut;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t sliceOffset;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t outputFullDim;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.loraIndices = loraIndices;
    __ascendc_args.loraIndicesSize = loraIndicesSize;
    __ascendc_args.seqLen = seqLen;
    __ascendc_args.seqLenSize = seqLenSize;
    __ascendc_args.yIn = yIn;
    __ascendc_args.yOut = yOut;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.outputHiddenDim = outputHiddenDim;
    __ascendc_args.sliceOffset = sliceOffset;
    __ascendc_args.outputFullDim = outputFullDim;

    __ascendc_ret = launch_and_profiling_sgmv_expand_half(6, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_sgmv_shrink_bfloat16_t(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "sgmv_shrink_bfloat16_t";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_sgmv_shrink_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* loraIndices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t loraIndicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* seqLen;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t seqLenSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t inputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(float) + 3) >> 2) << 2) float scale;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.loraIndices = loraIndices;
    __ascendc_args.loraIndicesSize = loraIndicesSize;
    __ascendc_args.seqLen = seqLen;
    __ascendc_args.seqLenSize = seqLenSize;
    __ascendc_args.y = y;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.inputHiddenDim = inputHiddenDim;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.scale = scale;

    __ascendc_ret = launch_and_profiling_sgmv_shrink_bfloat16_t(7, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_sgmv_shrink_half(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "sgmv_shrink_half";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_sgmv_shrink_half(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* weight;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* loraIndices;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t loraIndicesSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* seqLen;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t seqLenSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t numTokensPerCore;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t inputHiddenDim;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t maxLoRARank;
        alignas(((alignof(float) + 3) >> 2) << 2) float scale;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.weight = weight;
    __ascendc_args.loraIndices = loraIndices;
    __ascendc_args.loraIndicesSize = loraIndicesSize;
    __ascendc_args.seqLen = seqLen;
    __ascendc_args.seqLenSize = seqLenSize;
    __ascendc_args.y = y;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.numTokensPerCore = numTokensPerCore;
    __ascendc_args.inputHiddenDim = inputHiddenDim;
    __ascendc_args.maxLoRARank = maxLoRARank;
    __ascendc_args.scale = scale;

    __ascendc_ret = launch_and_profiling_sgmv_shrink_half(8, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_mla_preprocess(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "mla_preprocess";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 0, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_mla_preprocess(uint32_t blockDim, void* stream, void* hiddenState, void* quantScale1, void* quantOffset1, void* wdqkv, void* bias1, void* gamma2, void* beta2, void* quantScale2, void* quantOffset2, void* gamma3, void* sin1, void* cos1, void* sin2, void* cos2, void* keycache, void* slotMapping, void* wuq, void* bias2, void* wuk, void* descale1, void* descale2, void* ctkvScale, void* qnopeScale, void* q, void* keycacheOut, void* q2, void* keycacheOut2, void* innerOut, void* workspace, void* tiling)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* ffts_addr;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* hiddenState;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* quantScale1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* quantOffset1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* wdqkv;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* bias1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* gamma2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* beta2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* quantScale2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* quantOffset2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* gamma3;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* sin1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* cos1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* sin2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* cos2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* keycache;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* slotMapping;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* wuq;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* bias2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* wuk;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* descale1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* descale2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* ctkvScale;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* qnopeScale;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* q;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* keycacheOut;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* q2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* keycacheOut2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* innerOut;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* workspace;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* tiling;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);

    void *ffts_addr;
    __ascendc_ret = GetAscendCoreSyncAddr(&ffts_addr);
    if (__ascendc_ret != 0) {
        printf("GetAscendCoreSyncAddr ret %u\n", __ascendc_ret);
        return __ascendc_ret;
    }

    __ascendc_args.ffts_addr = ffts_addr;
    __ascendc_args.hiddenState = hiddenState;
    __ascendc_args.quantScale1 = quantScale1;
    __ascendc_args.quantOffset1 = quantOffset1;
    __ascendc_args.wdqkv = wdqkv;
    __ascendc_args.bias1 = bias1;
    __ascendc_args.gamma2 = gamma2;
    __ascendc_args.beta2 = beta2;
    __ascendc_args.quantScale2 = quantScale2;
    __ascendc_args.quantOffset2 = quantOffset2;
    __ascendc_args.gamma3 = gamma3;
    __ascendc_args.sin1 = sin1;
    __ascendc_args.cos1 = cos1;
    __ascendc_args.sin2 = sin2;
    __ascendc_args.cos2 = cos2;
    __ascendc_args.keycache = keycache;
    __ascendc_args.slotMapping = slotMapping;
    __ascendc_args.wuq = wuq;
    __ascendc_args.bias2 = bias2;
    __ascendc_args.wuk = wuk;
    __ascendc_args.descale1 = descale1;
    __ascendc_args.descale2 = descale2;
    __ascendc_args.ctkvScale = ctkvScale;
    __ascendc_args.qnopeScale = qnopeScale;
    __ascendc_args.q = q;
    __ascendc_args.keycacheOut = keycacheOut;
    __ascendc_args.q2 = q2;
    __ascendc_args.keycacheOut2 = keycacheOut2;
    __ascendc_args.innerOut = innerOut;
    __ascendc_args.workspace = workspace;
    __ascendc_args.tiling = tiling;

    __ascendc_ret = launch_and_profiling_mla_preprocess(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_batch_matmul_transpose(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "batch_matmul_transpose";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aic == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aic, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 6, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_batch_matmul_transpose(uint32_t blockDim, void* stream, void* gm_a, void* gm_b, void* gm_c, void* gm_tiling_data)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* gm_a;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* gm_b;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* gm_c;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* gm_tiling_data;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.gm_a = gm_a;
    __ascendc_args.gm_b = gm_b;
    __ascendc_args.gm_c = gm_c;
    __ascendc_args.gm_tiling_data = gm_tiling_data;

    __ascendc_ret = launch_and_profiling_batch_matmul_transpose(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
