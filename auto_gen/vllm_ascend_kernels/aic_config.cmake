set(MIX_SOURCES
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_mla_preprocess_kernel.cpp
)
set(AIC_SOURCES
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_batch_matmul_transpose_kernel.cpp
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_batch_matmul_transpose_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_batch_matmul_transpose_kernel=batch_matmul_transpose_0;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_mla_preprocess_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_mla_preprocess_kernel=mla_preprocess_0_mix_aic;ONE_CORE_DUMP_SIZE=1048576"
)
