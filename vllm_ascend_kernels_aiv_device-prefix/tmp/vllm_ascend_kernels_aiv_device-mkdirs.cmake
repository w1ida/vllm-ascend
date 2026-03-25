# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/local/Ascend/cann-8.5.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/device_project")
  file(MAKE_DIRECTORY "/usr/local/Ascend/cann-8.5.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/device_project")
endif()
file(MAKE_DIRECTORY
  "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/src/vllm_ascend_kernels_aiv_device-build"
  "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix"
  "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/tmp"
  "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/src/vllm_ascend_kernels_aiv_device-stamp"
  "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/src"
  "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/src/vllm_ascend_kernels_aiv_device-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/src/vllm_ascend_kernels_aiv_device-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/vllm-ascend/vllm_ascend_kernels_aiv_device-prefix/src/vllm_ascend_kernels_aiv_device-stamp${cfgdir}") # cfgdir has leading slash
endif()
