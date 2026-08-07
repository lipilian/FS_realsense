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
option(FFS_VIEWER_BUILD_SENTECH_DUAL_CONNECT
       "Build the Sentech dual-camera ImGui viewer" ON)
option(FFS_VIEWER_WARNINGS_AS_ERRORS
       "Treat C++ warnings in this project as errors" OFF)

set(FFS_TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation prefix")
set(FFS_VIEWER_DEFAULT_ENGINE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ffs_1280x800" CACHE PATH
    "Default directory containing fast_foundationstereo.engine and onnx.yaml")
set(FFS_VIEWER_DEFAULT_FS_ENGINE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/models/fs_960x608_iters32" CACHE PATH
    "Default directory containing the high-quality fs.engine")
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

# This target is deliberately independent of the RealSense/live-viewer stack.
# Camera roles are matched by their configured Sentech names, never by the
# discovery order reported by StApi.
if(FFS_VIEWER_BUILD_SENTECH_DUAL_CONNECT)
  set(FFS_SENTECH_ROOT "/opt/sentech" CACHE PATH
      "Sentech StApi SDK installation prefix")

  find_path(FFS_SENTECH_STAPI_INCLUDE_DIR StApi_TL.h
    PATHS "${FFS_SENTECH_ROOT}/include/StApi"
    NO_DEFAULT_PATH)
  find_path(FFS_SENTECH_GENICAM_INCLUDE_DIR GenApi/GenApi.h
    PATHS "${FFS_SENTECH_ROOT}/include/GenICam"
    NO_DEFAULT_PATH)
  find_library(FFS_SENTECH_TL_LIBRARY NAMES StApi_TL
    PATHS "${FFS_SENTECH_ROOT}/lib"
    NO_DEFAULT_PATH)
  find_library(FFS_SENTECH_GCBASE_LIBRARY NAMES GCBase
    PATHS "${FFS_SENTECH_ROOT}/lib/GenICam"
    NO_DEFAULT_PATH)
  find_library(FFS_SENTECH_GENAPI_LIBRARY NAMES GenApi
    PATHS "${FFS_SENTECH_ROOT}/lib/GenICam"
    NO_DEFAULT_PATH)
  find_library(FFS_SENTECH_IP_LIBRARY NAMES StApi_IP
    PATHS "${FFS_SENTECH_ROOT}/lib"
    NO_DEFAULT_PATH)
  find_library(FFS_SENTECH_TURBOJPEG_LIBRARY NAMES turbojpeg
    PATHS "${FFS_SENTECH_ROOT}/lib"
    NO_DEFAULT_PATH)

  if(NOT FFS_SENTECH_STAPI_INCLUDE_DIR OR NOT FFS_SENTECH_GENICAM_INCLUDE_DIR
     OR NOT FFS_SENTECH_TL_LIBRARY OR NOT FFS_SENTECH_GCBASE_LIBRARY
     OR NOT FFS_SENTECH_GENAPI_LIBRARY OR NOT FFS_SENTECH_IP_LIBRARY
     OR NOT FFS_SENTECH_TURBOJPEG_LIBRARY)
    message(FATAL_ERROR
      "Sentech StApi SDK was not found under ${FFS_SENTECH_ROOT}. "
      "Set FFS_SENTECH_ROOT to the SDK installation prefix or disable "
      "FFS_VIEWER_BUILD_SENTECH_DUAL_CONNECT.")
  endif()

  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc aruco calib3d)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
  find_package(GLEW REQUIRED)

  add_library(sentech_imgui STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(sentech_imgui PUBLIC
    third_party/imgui
    third_party/imgui/backends
  )
  target_link_libraries(sentech_imgui PUBLIC glfw OpenGL::GL)

  add_executable(sentech_stereo
    apps/sentech_stereo/main.cpp
    src/calibration/live_charuco_detector.cpp
    src/calibration/stereo_charuco_calibrator.cpp
    src/calibration/stereo_rectifier.cpp
    src/inference/ffs_runner.cpp
    src/inference/sentech_ffs_pipeline.cpp
    src/io/sentech_stereo_source.cpp
    src/ui/sentech_point_cloud_viewer.cpp
  )
  target_compile_features(sentech_stereo PRIVATE cxx_std_20)
  target_compile_definitions(sentech_stereo PRIVATE
    FFS_SENTECH_GENTL_DIRECTORY="${FFS_SENTECH_ROOT}/lib"
    FFS_SENTECH_FFS_ENGINE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models/ffs_608x512"
  )
  target_include_directories(sentech_stereo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${FFS_SENTECH_STAPI_INCLUDE_DIR}"
    "${FFS_SENTECH_GENICAM_INCLUDE_DIR}"
  )
  target_link_libraries(sentech_stereo PRIVATE
    "${FFS_SENTECH_TL_LIBRARY}"
    "${FFS_SENTECH_GCBASE_LIBRARY}"
    "${FFS_SENTECH_GENAPI_LIBRARY}"
    "${FFS_SENTECH_IP_LIBRARY}"
    "${FFS_SENTECH_TURBOJPEG_LIBRARY}"
    ${OpenCV_LIBS}
    ffs::runtime
    sentech_imgui
    glfw
    GLEW::GLEW
    OpenGL::GL
    Threads::Threads
  )
  set_target_properties(sentech_stereo PROPERTIES
    BUILD_RPATH "${FFS_SENTECH_ROOT}/lib;${FFS_SENTECH_ROOT}/lib/GenICam"
  )
  # StApi_IP has Sentech-private transitive dependencies (such as turbojpeg).
  # Use DT_RPATH so this SDK path is also searched for those dependencies.
  target_link_options(sentech_stereo PRIVATE
    "-Wl,--disable-new-dtags"
    "-Wl,--no-as-needed"
  )
  ffs_viewer_enable_warnings(sentech_stereo)
endif()

if(FFS_VIEWER_BUILD_OFFLINE_VALIDATE)
  # DB3 playback comes from the current librealsense recording format; OpenCV
  # is used for lossless Y8 PNG artifacts and disparity visualizations.
  find_package(realsense2 2.58 REQUIRED)
  find_package(OpenCV 4 QUIET COMPONENTS core highgui imgcodecs imgproc viz)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
      "offline_validate requires OpenCV 4 development files (core, imgcodecs, imgproc). "
      "On Ubuntu install libopencv-dev.")
  endif()
  add_library(ffs_viewer_imgui STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(ffs_viewer_imgui PUBLIC
    third_party/imgui
    third_party/imgui/backends
  )
  target_link_libraries(ffs_viewer_imgui PUBLIC glfw OpenGL::GL)

  add_library(ffs_viewer_offline STATIC
    src/io/db3_stereo_source.cpp
    src/io/d455_stereo_source.cpp
    src/inference/ffs_runner.cpp
    src/inference/fs_runner.cpp
    src/geometry/final_cloud_processor.cu
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

  add_executable(offline_validate apps/offline_validate/main.cpp)
  target_link_libraries(offline_validate PRIVATE ffs_viewer::offline)
  target_compile_definitions(offline_validate PRIVATE
    FFS_VIEWER_DEFAULT_ENGINE_DIR="${FFS_VIEWER_DEFAULT_ENGINE_DIR}"
    FFS_VIEWER_DEFAULT_CONFIG="${FFS_VIEWER_DEFAULT_CONFIG}"
  )
  ffs_viewer_enable_warnings(offline_validate)

  add_executable(live_infer apps/live_infer/main.cpp)
  target_link_libraries(live_infer PRIVATE ffs_viewer::offline)
  target_compile_definitions(live_infer PRIVATE
    FFS_VIEWER_DEFAULT_ENGINE_DIR="${FFS_VIEWER_DEFAULT_ENGINE_DIR}"
  )
  ffs_viewer_enable_warnings(live_infer)

  add_executable(live_viewer
    apps/live_viewer/main.cpp
    apps/live_viewer/live_pipeline.cpp
    apps/live_viewer/opengl_point_cloud_viewer.cpp
  )
  target_link_libraries(live_viewer PRIVATE ffs_viewer::offline ffs_viewer_imgui glfw GLEW OpenGL::GL)
  target_compile_definitions(live_viewer PRIVATE
    FFS_VIEWER_DEFAULT_ENGINE_DIR="${FFS_VIEWER_DEFAULT_ENGINE_DIR}"
    FFS_VIEWER_DEFAULT_FS_ENGINE_DIR="${FFS_VIEWER_DEFAULT_FS_ENGINE_DIR}"
  )
  ffs_viewer_enable_warnings(live_viewer)

  add_custom_command(TARGET offline_validate POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:offline_validate>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFS_VIEWER_DEFAULT_CONFIG}"
            "$<TARGET_FILE_DIR:offline_validate>/config/offline_validate.yaml"
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
