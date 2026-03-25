# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/local/Ascend/cann-8.5.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/device_preprocess_project")
  file(MAKE_DIRECTORY "/usr/local/Ascend/cann-8.5.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/device_preprocess_project")
endif()
file(MAKE_DIRECTORY
  "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-build"
  "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix"
  "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/tmp"
  "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-stamp"
  "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/src"
  "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/vllm-ascend/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-stamp${cfgdir}") # cfgdir has leading slash
endif()
