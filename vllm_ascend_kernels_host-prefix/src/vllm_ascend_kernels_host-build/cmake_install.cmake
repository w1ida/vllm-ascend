# Install script for directory: /usr/local/Ascend/cann-8.5.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/host_project

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/workspace/vllm-ascend/vllm_ascend_kernels_host_dir")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/kernels/bgmv_expand.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/kernels/bgmv_expand.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/kernels/bgmv_shrink.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/kernels/bgmv_shrink.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/kernels/get_masked_input_and_mask_kernel.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/kernels/get_masked_input_and_mask_kernel.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/kernels/sgmv_expand.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/kernels/sgmv_expand.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/kernels/sgmv_shrink.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/kernels/sgmv_shrink.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp.o")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./objects-Release/host_bisheng_obj" TYPE FILE RENAME "workspace/vllm-ascend/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp.o" FILES "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir//workspace/vllm-ascend/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp.o")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/CMakeFiles/host_bisheng_obj.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/workspace/vllm-ascend/vllm_ascend_kernels_host-prefix/src/vllm_ascend_kernels_host-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
