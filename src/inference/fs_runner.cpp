#include "ffs_viewer/inference/fs_runner.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ffs_viewer::inference {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[FS TensorRT] " << message << '\n';
        }
    }
};

TrtLogger g_logger;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void allocate(void** pointer, std::size_t bytes, const char* operation) {
    checkCuda(cudaMalloc(pointer, bytes), operation);
}

float elapsedMilliseconds(cudaEvent_t begin, cudaEvent_t end, const char* operation) {
    float elapsed_ms = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, begin, end), operation);
    return elapsed_ms;
}

std::filesystem::path resolveEnginePath(const std::string& path_string) {
    const std::filesystem::path path(path_string);
    return std::filesystem::is_directory(path) ? path / "fs.engine" : path;
}

void requireTensor(const nvinfer1::ICudaEngine& engine, const char* name,
                   nvinfer1::TensorIOMode expected_mode) {
    if (engine.getTensorIOMode(name) != expected_mode) {
        const char* kind = expected_mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output";
        throw std::runtime_error(std::string("FS engine is missing expected ") + kind + ": " + name);
    }
}


void packY8AsPaddedRgb(const std::vector<std::uint8_t>& source, int source_width, int source_height,
                       int model_width, int content_height, int pad_top, int pad_bottom,
                       std::vector<float>& output) {
    const std::size_t source_pixels = static_cast<std::size_t>(source_width) * source_height;
    if (source.size() != source_pixels) {
        throw std::runtime_error("Stereo frame does not contain a complete Y8 image");
    }

    const cv::Mat source_image(source_height, source_width, CV_8UC1,
                               const_cast<std::uint8_t*>(source.data()));
    cv::Mat resized;
    cv::resize(source_image, resized, cv::Size(model_width, content_height), 0.0, 0.0,
               cv::INTER_LINEAR);

    const int model_height = content_height + pad_top + pad_bottom;
    const std::size_t pixels = static_cast<std::size_t>(model_width) * model_height;
    output.resize(3 * pixels);
    for (int y = 0; y < model_height; ++y) {
        const int content_y = std::clamp(y - pad_top, 0, content_height - 1);
        const auto* source_row = resized.ptr<std::uint8_t>(content_y);
        for (int x = 0; x < model_width; ++x) {
            const float value = static_cast<float>(source_row[x]);
            const std::size_t index = static_cast<std::size_t>(y) * model_width + x;
            output[index] = value;
            output[pixels + index] = value;
            output[2 * pixels + index] = value;
        }
    }
}

void packBgrAsPaddedChw(const std::vector<std::uint8_t>& source, int source_width, int source_height,
                        int model_width, int content_height, int pad_top, int pad_bottom,
                        std::vector<float>& output) {
    const std::size_t source_bytes = static_cast<std::size_t>(source_width) * source_height * 3U;
    if (source.size() != source_bytes)
        throw std::runtime_error("Stereo frame does not contain a complete BGR8 image");

    const cv::Mat source_image(source_height, source_width, CV_8UC3,
                               const_cast<std::uint8_t*>(source.data()));
    cv::Mat resized;
    cv::resize(source_image, resized, cv::Size(model_width, content_height), 0.0, 0.0,
               cv::INTER_LINEAR);

    const int model_height = content_height + pad_top + pad_bottom;
    const std::size_t pixels = static_cast<std::size_t>(model_width) * model_height;
    output.resize(3 * pixels);
    for (int y = 0; y < model_height; ++y) {
        const int content_y = std::clamp(y - pad_top, 0, content_height - 1);
        const cv::Vec3b* source_row = resized.ptr<cv::Vec3b>(content_y);
        for (int x = 0; x < model_width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * model_width + x;
            // FoundationStereo training/demo input is OpenCV BGR stored as CHW.
            output[index] = static_cast<float>(source_row[x][0]);
            output[pixels + index] = static_cast<float>(source_row[x][1]);
            output[2 * pixels + index] = static_cast<float>(source_row[x][2]);
        }
    }
}

}  // namespace

struct FsRunner::Impl {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    float* d_left = nullptr;
    float* d_right = nullptr;
    float* d_disparity = nullptr;
    cudaEvent_t timing_begin = nullptr;
    cudaEvent_t timing_h2d_done = nullptr;
    cudaEvent_t timing_inference_done = nullptr;
    cudaEvent_t timing_end = nullptr;
    int model_width = 0;
    int model_height = 0;
    int content_height = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    std::vector<float> host_left;
    std::vector<float> host_right;
    std::vector<float> host_padded_disparity;
    std::unique_ptr<geometry::FinalCloudProcessor> final_cloud_processor;

    ~Impl() {
        if (timing_begin != nullptr) cudaEventDestroy(timing_begin);
        if (timing_h2d_done != nullptr) cudaEventDestroy(timing_h2d_done);
        if (timing_inference_done != nullptr) cudaEventDestroy(timing_inference_done);
        if (timing_end != nullptr) cudaEventDestroy(timing_end);
        if (d_left != nullptr) cudaFree(d_left);
        if (d_right != nullptr) cudaFree(d_right);
        if (d_disparity != nullptr) cudaFree(d_disparity);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

FsRunner::FsRunner(std::string engine_path) : impl_(std::make_unique<Impl>()) {
    const std::filesystem::path resolved_path = resolveEnginePath(engine_path);
    std::ifstream file(resolved_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open FS TensorRT engine: " + resolved_path.string());
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("FS TensorRT engine is empty: " + resolved_path.string());
    }
    std::vector<char> serialized(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(serialized.data(), size)) {
        throw std::runtime_error("Cannot read FS TensorRT engine: " + resolved_path.string());
    }

    // TensorRT accepts one process-wide logger. The live FFS runner has already
    // registered its logger, so reuse it when available instead of registering
    // this adapter's fallback logger and emitting a warning.
    nvinfer1::ILogger* logger = getLogger();
    if (logger == nullptr) logger = &g_logger;
    impl_->runtime.reset(nvinfer1::createInferRuntime(*logger));
    if (!impl_->runtime) {
        throw std::runtime_error("TensorRT failed to create the FS runtime");
    }
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!impl_->engine) {
        throw std::runtime_error("TensorRT failed to deserialize FS engine: " + resolved_path.string());
    }
    requireTensor(*impl_->engine, "left", nvinfer1::TensorIOMode::kINPUT);
    requireTensor(*impl_->engine, "right", nvinfer1::TensorIOMode::kINPUT);
    requireTensor(*impl_->engine, "disp", nvinfer1::TensorIOMode::kOUTPUT);

    const nvinfer1::Dims left_dimensions = impl_->engine->getTensorShape("left");
    const nvinfer1::Dims right_dimensions = impl_->engine->getTensorShape("right");
    const nvinfer1::Dims output_dimensions = impl_->engine->getTensorShape("disp");
    if (left_dimensions.nbDims != 4 || left_dimensions.d[0] != 1 || left_dimensions.d[1] != 3 ||
        right_dimensions.nbDims != 4 || right_dimensions.d[0] != 1 || right_dimensions.d[1] != 3 ||
        output_dimensions.nbDims != 4 || output_dimensions.d[0] != 1 || output_dimensions.d[1] != 1 ||
        left_dimensions.d[2] != right_dimensions.d[2] || left_dimensions.d[3] != right_dimensions.d[3] ||
        left_dimensions.d[2] != output_dimensions.d[2] || left_dimensions.d[3] != output_dimensions.d[3]) {
        throw std::runtime_error("Unexpected FS input/output tensor shapes");
    }
    if (impl_->engine->getTensorDataType("left") != nvinfer1::DataType::kFLOAT ||
        impl_->engine->getTensorDataType("right") != nvinfer1::DataType::kFLOAT ||
        impl_->engine->getTensorDataType("disp") != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("FS adapter requires FP32 left, right, and disp tensors");
    }

    impl_->model_width = left_dimensions.d[3];
    impl_->model_height = left_dimensions.d[2];
    // The legacy D455 model is 960x608 with an artificial 4-row pad on
    // each side. Sentech's 608x512 model is natively divisible by 32.
    if (impl_->model_width == 960 && impl_->model_height == 608) {
        impl_->pad_top = 4;
        impl_->pad_bottom = 4;
    }
    impl_->content_height = impl_->model_height - impl_->pad_top - impl_->pad_bottom;
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) {
        throw std::runtime_error("TensorRT failed to create the FS execution context");
    }
    checkCuda(cudaStreamCreate(&impl_->stream), "create FS CUDA stream");

    const std::size_t pixels = static_cast<std::size_t>(impl_->model_width) * impl_->model_height;
    allocate(reinterpret_cast<void**>(&impl_->d_left), 3 * pixels * sizeof(float), "allocate FS left input");
    allocate(reinterpret_cast<void**>(&impl_->d_right), 3 * pixels * sizeof(float), "allocate FS right input");
    allocate(reinterpret_cast<void**>(&impl_->d_disparity), pixels * sizeof(float), "allocate FS disparity output");
    impl_->final_cloud_processor = std::make_unique<geometry::FinalCloudProcessor>(impl_->model_width,
                                                                                     impl_->content_height);
    if (!impl_->context->setTensorAddress("left", impl_->d_left) ||
        !impl_->context->setTensorAddress("right", impl_->d_right) ||
        !impl_->context->setTensorAddress("disp", impl_->d_disparity)) {
        throw std::runtime_error("TensorRT failed to bind FS input or output tensors");
    }
    checkCuda(cudaEventCreate(&impl_->timing_begin), "create FS timing begin event");
    checkCuda(cudaEventCreate(&impl_->timing_h2d_done), "create FS timing H2D event");
    checkCuda(cudaEventCreate(&impl_->timing_inference_done), "create FS timing inference event");
    checkCuda(cudaEventCreate(&impl_->timing_end), "create FS timing end event");
}

FsRunner::~FsRunner() = default;

DisparityFrame FsRunner::infer(const io::StereoFrame& stereo) {
    if (stereo.width <= 0 || stereo.height <= 0) {
        throw std::runtime_error("FS inference requires positive stereo dimensions");
    }
    packY8AsPaddedRgb(stereo.left_y8, stereo.width, stereo.height, impl_->model_width,
                       impl_->content_height, impl_->pad_top, impl_->pad_bottom, impl_->host_left);
    packY8AsPaddedRgb(stereo.right_y8, stereo.width, stereo.height, impl_->model_width,
                       impl_->content_height, impl_->pad_top, impl_->pad_bottom, impl_->host_right);

    const std::size_t model_pixels = static_cast<std::size_t>(impl_->model_width) * impl_->model_height;
    const auto host_begin = std::chrono::steady_clock::now();
    checkCuda(cudaEventRecord(impl_->timing_begin, impl_->stream), "record FS timing begin");
    checkCuda(cudaMemcpyAsync(impl_->d_left, impl_->host_left.data(),
                              impl_->host_left.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream),
              "copy FS left input to GPU");
    checkCuda(cudaMemcpyAsync(impl_->d_right, impl_->host_right.data(),
                              impl_->host_right.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream),
              "copy FS right input to GPU");
    checkCuda(cudaEventRecord(impl_->timing_h2d_done, impl_->stream), "record FS H2D completion");
    if (!impl_->context->enqueueV3(impl_->stream)) {
        throw std::runtime_error("TensorRT failed to enqueue FS inference");
    }
    checkCuda(cudaEventRecord(impl_->timing_inference_done, impl_->stream),
              "record FS inference completion");

    impl_->host_padded_disparity.resize(model_pixels);
    checkCuda(cudaMemcpyAsync(impl_->host_padded_disparity.data(), impl_->d_disparity,
                              model_pixels * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream),
              "copy FS disparity from GPU");
    checkCuda(cudaEventRecord(impl_->timing_end, impl_->stream), "record FS timing end");
    checkCuda(cudaStreamSynchronize(impl_->stream), "synchronize FS inference stream");
    const auto host_end = std::chrono::steady_clock::now();

    DisparityFrame output;
    output.width = impl_->model_width;
    output.height = impl_->content_height;
    output.values.resize(static_cast<std::size_t>(output.width) * output.height);
    for (int y = 0; y < output.height; ++y) {
        const float* source = impl_->host_padded_disparity.data() +
                              static_cast<std::size_t>(y + impl_->pad_top) * output.width;
        float* destination = output.values.data() + static_cast<std::size_t>(y) * output.width;
        std::copy_n(source, output.width, destination);
    }
    output.timing.h2d_ms = elapsedMilliseconds(impl_->timing_begin, impl_->timing_h2d_done,
                                               "measure FS H2D time");
    output.timing.inference_ms = elapsedMilliseconds(impl_->timing_h2d_done,
                                                     impl_->timing_inference_done,
                                                     "measure FS inference time");
    output.timing.d2h_ms = elapsedMilliseconds(impl_->timing_inference_done, impl_->timing_end,
                                               "measure FS D2H time");
    output.timing.gpu_total_ms = elapsedMilliseconds(impl_->timing_begin, impl_->timing_end,
                                                     "measure FS GPU total time");
    output.timing.host_total_ms =
        std::chrono::duration<float, std::milli>(host_end - host_begin).count();
    return output;
}

geometry::FinalCloudFrame FsRunner::inferFinal(const io::StereoFrame& stereo,
                                                const io::StereoCalibration& calibration,
                                                float z_max_m,
                                                const std::function<void()>& on_denoise) {
    if (stereo.width <= 0 || stereo.height <= 0) {
        throw std::runtime_error("FS final inference requires positive stereo dimensions");
    }
    packY8AsPaddedRgb(stereo.left_y8, stereo.width, stereo.height, impl_->model_width,
                       impl_->content_height, impl_->pad_top, impl_->pad_bottom, impl_->host_left);
    packY8AsPaddedRgb(stereo.right_y8, stereo.width, stereo.height, impl_->model_width,
                       impl_->content_height, impl_->pad_top, impl_->pad_bottom, impl_->host_right);
    checkCuda(cudaMemcpyAsync(impl_->d_left, impl_->host_left.data(), impl_->host_left.size() * sizeof(float),
                              cudaMemcpyHostToDevice, impl_->stream), "copy final FS left input to GPU");
    checkCuda(cudaMemcpyAsync(impl_->d_right, impl_->host_right.data(), impl_->host_right.size() * sizeof(float),
                              cudaMemcpyHostToDevice, impl_->stream), "copy final FS right input to GPU");
    if (!impl_->context->enqueueV3(impl_->stream)) {
        throw std::runtime_error("TensorRT failed to enqueue final FS inference");
    }
    return impl_->final_cloud_processor->process(impl_->d_disparity, impl_->d_left, impl_->stream, calibration,
                                                  stereo.width, stereo.height,
                                                  impl_->pad_top * impl_->model_width, z_max_m, on_denoise);
}

geometry::FinalCloudFrame FsRunner::inferFinalBgr(
    int width, int height, const std::vector<std::uint8_t>& left_bgr,
    const std::vector<std::uint8_t>& right_bgr, const io::StereoCalibration& calibration,
    float z_max_m, const std::function<void()>& on_denoise) {
    if (width <= 0 || height <= 0)
        throw std::runtime_error("FS BGR inference requires positive stereo dimensions");
    packBgrAsPaddedChw(left_bgr, width, height, impl_->model_width, impl_->content_height,
                        impl_->pad_top, impl_->pad_bottom, impl_->host_left);
    packBgrAsPaddedChw(right_bgr, width, height, impl_->model_width, impl_->content_height,
                        impl_->pad_top, impl_->pad_bottom, impl_->host_right);
    checkCuda(cudaMemcpyAsync(impl_->d_left, impl_->host_left.data(),
                              impl_->host_left.size() * sizeof(float), cudaMemcpyHostToDevice,
                              impl_->stream), "copy final FS BGR left input to GPU");
    checkCuda(cudaMemcpyAsync(impl_->d_right, impl_->host_right.data(),
                              impl_->host_right.size() * sizeof(float), cudaMemcpyHostToDevice,
                              impl_->stream), "copy final FS BGR right input to GPU");
    if (!impl_->context->enqueueV3(impl_->stream))
        throw std::runtime_error("TensorRT failed to enqueue final FS BGR inference");
    return impl_->final_cloud_processor->process(impl_->d_disparity, impl_->d_left, impl_->stream,
                                                  calibration, width, height,
                                                  impl_->pad_top * impl_->model_width, z_max_m, on_denoise);
}

int FsRunner::modelWidth() const { return impl_->model_width; }
int FsRunner::modelHeight() const { return impl_->model_height; }
int FsRunner::contentWidth() const { return impl_->model_width; }
int FsRunner::contentHeight() const { return impl_->content_height; }
int FsRunner::padTop() const { return impl_->pad_top; }
int FsRunner::padBottom() const { return impl_->pad_bottom; }

}  // namespace ffs_viewer::inference
