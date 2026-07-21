#include "ffs_viewer/inference/ffs_runner.hpp"

#include "ffs_depth_single_tensorrt.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ffs_viewer::inference {
namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

void allocate(void** pointer, std::size_t bytes, const char* name) {
    checkCuda(cudaMalloc(pointer, bytes), name);
}

float elapsedMilliseconds(cudaEvent_t begin, cudaEvent_t end, const char* operation) {
    float elapsed_ms = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, begin, end), operation);
    return elapsed_ms;
}

}  // namespace

struct FfsRunner::Impl {
    std::unique_ptr<ffs_depth::FFSSingleEngineInference> engine;
    std::uint8_t* d_left_y8 = nullptr;
    std::uint8_t* d_right_y8 = nullptr;
    float* d_disparity = nullptr;
    cudaEvent_t timing_begin = nullptr;
    cudaEvent_t timing_h2d_done = nullptr;
    cudaEvent_t timing_inference_done = nullptr;
    cudaEvent_t timing_end = nullptr;
    int width = 0;
    int height = 0;
    int max_disparity = 0;

    ~Impl() {
        if (timing_begin != nullptr) cudaEventDestroy(timing_begin);
        if (timing_h2d_done != nullptr) cudaEventDestroy(timing_h2d_done);
        if (timing_inference_done != nullptr) cudaEventDestroy(timing_inference_done);
        if (timing_end != nullptr) cudaEventDestroy(timing_end);
        if (d_left_y8 != nullptr) cudaFree(d_left_y8);
        if (d_right_y8 != nullptr) cudaFree(d_right_y8);
        if (d_disparity != nullptr) cudaFree(d_disparity);
    }
};

FfsRunner::FfsRunner(std::string engine_dir) : impl_(std::make_unique<Impl>()) {
    impl_->engine = std::make_unique<ffs_depth::FFSSingleEngineInference>(engine_dir);
    impl_->width = impl_->engine->modelWidth();
    impl_->height = impl_->engine->modelHeight();
    impl_->max_disparity = impl_->engine->maxDisp();
    if (impl_->width <= 0 || impl_->height <= 0 || impl_->max_disparity <= 0) {
        throw std::runtime_error("FFS engine reported an invalid input configuration");
    }

    const std::size_t pixels = static_cast<std::size_t>(impl_->width) * impl_->height;
    allocate(reinterpret_cast<void**>(&impl_->d_left_y8), pixels, "cudaMalloc left Y8 buffer");
    allocate(reinterpret_cast<void**>(&impl_->d_right_y8), pixels, "cudaMalloc right Y8 buffer");
    allocate(reinterpret_cast<void**>(&impl_->d_disparity),
             pixels * sizeof(float), "cudaMalloc disparity buffer");
    checkCuda(cudaEventCreate(&impl_->timing_begin), "create benchmark start event");
    checkCuda(cudaEventCreate(&impl_->timing_h2d_done), "create benchmark H2D event");
    checkCuda(cudaEventCreate(&impl_->timing_inference_done), "create benchmark inference event");
    checkCuda(cudaEventCreate(&impl_->timing_end), "create benchmark end event");
}

FfsRunner::~FfsRunner() = default;

DisparityFrame FfsRunner::infer(const io::StereoFrame& stereo) {
    if (stereo.width != impl_->width || stereo.height != impl_->height) {
        throw std::runtime_error(
            "D455 frame dimensions do not match this fixed-resolution FFS engine");
    }

    const std::size_t pixels = static_cast<std::size_t>(stereo.width) * stereo.height;
    if (stereo.left_y8.size() != pixels || stereo.right_y8.size() != pixels) {
        throw std::runtime_error("Stereo frame does not contain a complete Y8 image pair");
    }

    const cudaStream_t stream = impl_->engine->stream();
    const auto host_begin = std::chrono::steady_clock::now();
    checkCuda(cudaEventRecord(impl_->timing_begin, stream), "record benchmark start event");
    checkCuda(cudaMemcpyAsync(impl_->d_left_y8, stereo.left_y8.data(), pixels,
                              cudaMemcpyHostToDevice, stream),
              "copy left Y8 frame to GPU");
    checkCuda(cudaMemcpyAsync(impl_->d_right_y8, stereo.right_y8.data(), pixels,
                              cudaMemcpyHostToDevice, stream),
              "copy right Y8 frame to GPU");
    checkCuda(cudaEventRecord(impl_->timing_h2d_done, stream), "record benchmark H2D event");

    impl_->engine->inferY8(impl_->d_left_y8, impl_->d_right_y8,
                           stereo.height, stereo.width, impl_->d_disparity);
    checkCuda(cudaEventRecord(impl_->timing_inference_done, stream),
              "record benchmark inference event");

    DisparityFrame disparity;
    disparity.width = stereo.width;
    disparity.height = stereo.height;
    disparity.values.resize(pixels);
    checkCuda(cudaMemcpyAsync(disparity.values.data(), impl_->d_disparity,
                              pixels * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "copy disparity frame from GPU");
    checkCuda(cudaEventRecord(impl_->timing_end, stream), "record benchmark end event");
    checkCuda(cudaStreamSynchronize(stream), "synchronize FFS inference stream");
    const auto host_end = std::chrono::steady_clock::now();

    disparity.timing.h2d_ms = elapsedMilliseconds(
        impl_->timing_begin, impl_->timing_h2d_done, "measure benchmark H2D time");
    disparity.timing.inference_ms = elapsedMilliseconds(
        impl_->timing_h2d_done, impl_->timing_inference_done, "measure benchmark inference time");
    disparity.timing.d2h_ms = elapsedMilliseconds(
        impl_->timing_inference_done, impl_->timing_end, "measure benchmark D2H time");
    disparity.timing.gpu_total_ms = elapsedMilliseconds(
        impl_->timing_begin, impl_->timing_end, "measure benchmark GPU total time");
    disparity.timing.host_total_ms =
        std::chrono::duration<float, std::milli>(host_end - host_begin).count();
    return disparity;
}

int FfsRunner::modelWidth() const {
    return impl_->width;
}

int FfsRunner::modelHeight() const {
    return impl_->height;
}

int FfsRunner::maxDisparity() const {
    return impl_->max_disparity;
}

}  // namespace ffs_viewer::inference
