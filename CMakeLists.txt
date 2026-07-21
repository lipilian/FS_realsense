cmake_minimum_required(VERSION 3.24)

# The deployed RTX PRO 4000 Blackwell GPU is SM 120. A caller may override
# CMAKE_CUDA_ARCHITECTURES for a different deployment target.
set(FFS_VIEWER_CUDA_ARCHITECTURES "120" CACHE STRING
    "CUDA architectures used when CMAKE_CUDA_ARCHITECTURES is not set")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${FFS_VIEWER_CUDA_ARCHITECTURES}" CACHE STRING
      "CUDA architectures" FORCE)
endif()

project(ffs_viewer VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FFS_VIEWER_BUILD_OFFLINE_VALIDATE
       "Build the DB3-based offline validation executable" ON)
option(FFS_VIEWER_WARNINGS_AS_ERRORS
       "Treat C++ warnings in this project as errors" OFF)

set(FFS_TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation prefix")
set(FFS_VIEWER_DEFAULT_ENGINE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ffs_1280x800" CACHE PATH
    "Default directory containing fast_foundationstereo.engine and onnx.yaml")
set(FFS_VIEWER_DEFAULT_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/config/offline_validate.yaml" CACHE FILEPATH
    "Default offline-validation configuration file")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Route A: the vendored single-engine FFS runtime and FFSGWCVolume plugin.
add_subdirectory(third_party/fast_foundation_stereo_runtime)
if(NOT TARGET ffs::runtime)
  message(FATAL_ERROR "Vendored Fast-FoundationStereo runtime did not export ffs::runtime")
endif()

function(ffs_viewer_enable_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
  if(FFS_VIEWER_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
  endif()
endfunction()

if(FFS_VIEWER_BUILD_OFFLINE_VALIDATE)
  # DB3 playback comes from the current librealsense recording format; OpenCV
  # is used for lossless Y8 PNG artifacts and disparity visualizations.
  find_package(realsense2 2.58 REQUIRED)
  find_package(OpenCV 4 QUIET COMPONENTS core imgcodecs imgproc)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
      "ffs_offline_validate requires OpenCV 4 development files (core, imgcodecs, imgproc). "
      "On Ubuntu install libopencv-dev. To configure only the FFS runtime, use -DFFS_VIEWER_BUILD_OFFLINE_VALIDATE=OFF.")
  endif()

  add_library(ffs_viewer_offline STATIC
    src/io/db3_stereo_source.cpp
    src/inference/ffs_runner.cpp
    src/validation/offline_validator.cpp
  )
  add_library(ffs_viewer::offline ALIAS ffs_viewer_offline)

  target_compile_features(ffs_viewer_offline PUBLIC cxx_std_20)
  target_include_directories(ffs_viewer_offline
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(ffs_viewer_offline
    PUBLIC
      ffs::runtime
      realsense2::realsense2
      ${OpenCV_LIBS}
      Threads::Threads
  )
  set_target_properties(ffs_viewer_offline PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  ffs_viewer_enable_warnings(ffs_viewer_offline)

  add_executable(ffs_offline_validate apps/offline_validate/main.cpp)
  target_link_libraries(ffs_offline_validate PRIVATE ffs_viewer::offline)
  target_compile_definitions(ffs_offline_validate PRIVATE
    FFS_VIEWER_DEFAULT_ENGINE_DIR="${FFS_VIEWER_DEFAULT_ENGINE_DIR}"
    FFS_VIEWER_DEFAULT_CONFIG="${FFS_VIEWER_DEFAULT_CONFIG}"
  )
  ffs_viewer_enable_warnings(ffs_offline_validate)

  add_custom_command(TARGET ffs_offline_validate POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:ffs_offline_validate>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFS_VIEWER_DEFAULT_CONFIG}"
            "$<TARGET_FILE_DIR:ffs_offline_validate>/config/offline_validate.yaml"
    VERBATIM
  )
endif()

if(BUILD_TESTING)
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    message(STATUS "Catch2 found; unit and integration test targets can be added under tests/")
  else()
    message(STATUS "Catch2 3 not found; tests will be enabled after installing Catch2")
  endif()
endif()

if(EXISTS "${FFS_VIEWER_DEFAULT_ENGINE_DIR}/fast_foundationstereo.engine")
  message(STATUS "Default FFS engine: ${FFS_VIEWER_DEFAULT_ENGINE_DIR}")
else()
  message(STATUS "Default FFS engine is not present: ${FFS_VIEWER_DEFAULT_ENGINE_DIR}")
endif()
message(STATUS "FFS viewer CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
