set(MIX_SOURCES
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_mla_preprocess_kernel.cpp
)
set(AIV_SOURCES
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_bgmv_expand.cpp
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_bgmv_shrink.cpp
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_get_masked_input_and_mask_kernel.cpp
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_sgmv_expand.cpp
    /workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_sgmv_shrink.cpp
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_bgmv_expand.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_bgmv_expand_bfloat16_t_kernel=bgmv_expand_bfloat16_t_0;ONE_CORE_DUMP_SIZE=1048576;;auto_gen_bgmv_expand_half_kernel=bgmv_expand_half_1;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_bgmv_shrink.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_bgmv_shrink_bfloat16_t_kernel=bgmv_shrink_bfloat16_t_2;ONE_CORE_DUMP_SIZE=1048576;;auto_gen_bgmv_shrink_half_kernel=bgmv_shrink_half_3;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_get_masked_input_and_mask_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_get_masked_input_and_mask_kernel_kernel=get_masked_input_and_mask_kernel_4;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_mla_preprocess_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_mla_preprocess_kernel=mla_preprocess_0_mix_aiv;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_sgmv_expand.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_sgmv_expand_bfloat16_t_kernel=sgmv_expand_bfloat16_t_5;ONE_CORE_DUMP_SIZE=1048576;;auto_gen_sgmv_expand_half_kernel=sgmv_expand_half_6;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/auto_gen/vllm_ascend_kernels/auto_gen_sgmv_shrink.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_sgmv_shrink_bfloat16_t_kernel=sgmv_shrink_bfloat16_t_7;ONE_CORE_DUMP_SIZE=1048576;;auto_gen_sgmv_shrink_half_kernel=sgmv_shrink_half_8;ONE_CORE_DUMP_SIZE=1048576"
)
