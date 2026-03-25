add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/workspace/vllm-ascend/ascendc_runtime.cpp.o;/workspace/vllm-ascend/aicpu_rt.cpp.o;/workspace/vllm-ascend/ascendc_elf_tool.c.o"
)
