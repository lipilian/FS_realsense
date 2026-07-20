cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ffs_runtime LANGUAGES CXX CUDA)
elseif(NOT CMAKE_CUDA_COMPILER_LOADED)
  enable_language(CUDA)
endif()

find_package(CUDAToolkit REQUIRED)

set(FFS_TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation prefix")
set(_ffs_tensorrt_hints
  "${FFS_TENSORRT_ROOT}"
  /usr
  /usr/local
  /usr/local/tensorrt
  "${CUDAToolkit_TARGET_DIR}"
)

find_path(FFS_TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${_ffs_tensorrt_hints}
  PATH_SUFFIXES include include/x86_64-linux-gnu
)
find_library(FFS_NVINFER_LIBRARY
  NAMES nvinfer nvinfer.so.10
  HINTS ${_ffs_tensorrt_hints}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(NOT FFS_TENSORRT_INCLUDE_DIR OR NOT FFS_NVINFER_LIBRARY)
  message(FATAL_ERROR
    "TensorRT 10 headers/library not found. Set FFS_TENSORRT_ROOT to its prefix.")
endif()

add_library(ffs_runtime STATIC
  src/ffs_depth_single_tensorrt.cpp
  src/gwc_volume_plugin.cpp
  src/depth_kernels.cu
)
add_library(ffs::runtime ALIAS ffs_runtime)

target_compile_features(ffs_runtime PUBLIC cxx_std_17)
target_compile_options(ffs_runtime PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20013>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20011>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20091>
)
target_include_directories(ffs_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "${FFS_TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(ffs_runtime PUBLIC
  "${FFS_NVINFER_LIBRARY}"
  CUDA::cudart
)
set_target_properties(ffs_runtime PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)
