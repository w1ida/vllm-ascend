set_source_files_properties(/workspace/vllm-ascend/csrc/kernels/bgmv_expand.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/csrc/kernels/bgmv_shrink.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/csrc/kernels/get_masked_input_and_mask_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/csrc/kernels/sgmv_expand.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/csrc/kernels/sgmv_shrink.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/workspace/vllm-ascend/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS "ONE_CORE_DUMP_SIZE=1048576"
)
