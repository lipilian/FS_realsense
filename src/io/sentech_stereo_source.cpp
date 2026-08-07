#include "ffs_viewer/io/sentech_stereo_source.hpp"

#include <StApi_IP.h>
#include <StApi_TL.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace ffs_viewer::io {
namespace {

constexpr std::size_t kCameraCount = 2;
constexpr std::size_t kLeftCameraIndex = 0;
constexpr std::size_t kRightCameraIndex = 1;
constexpr std::array<const char *, kCameraCount> kCameraLabels{"Left camera", "Right camera"};
constexpr const char *kLeftCameraName = "STC-MCS500U3V(21LJ530)";
constexpr const char *kRightCameraName = "STC-MCS500U3V(21LJ548)";
constexpr std::size_t kLatestFramePoolSize = 3;

class StereoCapture {
  public:
    void start() {
        if (running_.load())
            return;

        try {
            worker_failed_.store(false);
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                latest_frames_.fill(nullptr);
            }
            display_frames_.fill(nullptr);
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                worker_error_.clear();
            }

            system_.Reset(StApi::CreateIStSystem());
            for (std::size_t discovered = 0; discovered < kCameraCount; ++discovered) {
                StApi::CIStDevicePtr device(system_->CreateFirstIStDevice());
                const std::size_t index = cameraIndexForName(*device->GetIStDeviceInfo());
                if (devices_[index].IsValid())
                    throw std::runtime_error("Both discovered cameras match " +
                                             std::string(kCameraLabels[index]));
                devices_[index].Reset(device.Move());
            }

            for (std::size_t index = 0; index < kCameraCount; ++index) {
                streams_[index].Reset(devices_[index]->CreateIStDataStream(0));
                converters_[index].Reset(
                    StApi::CreateIStConverter(StApi::StConverterType_PixelFormat));
                converters_[index]->SetDestinationPixelFormat(StApi::StPFNC_BGR8);
                converted_images_[index].Reset(StApi::CreateIStImageBuffer());

                std::cout << kCameraLabels[index] << " connected: "
                          << devices_[index]->GetIStDeviceInfo()->GetDisplayName() << '\n';
                streams_[index]->StartAcquisition();
                devices_[index]->AcquisitionStart();
            }

            running_.store(true);
            for (std::size_t index = 0; index < kCameraCount; ++index)
                workers_[index] = std::thread(&StereoCapture::captureLatestFrames, this, index);
            status_ = "Streaming two Sentech cameras (latest-frame buffering)";
        } catch (...) {
            stopNoThrow();
            throw;
        }
    }

    void stop() {
        stopNoThrow();
        status_ = "Stopped";
    }

    void poll() {
        if (worker_failed_.load()) {
            const std::string error = workerError();
            stopNoThrow();
            status_ = "Streaming error: " + error;
            return;
        }
        if (!running_.load())
            return;

        std::lock_guard<std::mutex> lock(frame_mutex_);
        display_frames_ = latest_frames_;
    }

    bool running() const {
        return running_.load();
    }

    const std::string &status() const {
        return status_;
    }

    const BgrFrame &leftFrame() const {
        return frameForIndex(kLeftCameraIndex);
    }

    const BgrFrame &rightFrame() const {
        return frameForIndex(kRightCameraIndex);
    }

    ~StereoCapture() {
        stopNoThrow();
    }

  private:
    static std::size_t cameraIndexForName(const StApi::IStDeviceInfo &info) {
        const std::string display_name(info.GetDisplayName().c_str());
        const std::string user_defined_name(info.GetUserDefinedName().c_str());
        const auto matches = [&](const char *configured_name) {
            return display_name == configured_name || user_defined_name == configured_name;
        };

        if (matches(kLeftCameraName))
            return kLeftCameraIndex;
        if (matches(kRightCameraName))
            return kRightCameraIndex;

        throw std::runtime_error("Unknown Sentech camera. DisplayName=\"" + display_name +
                                 "\", UserDefinedName=\"" + user_defined_name + "\"");
    }

    static void copyBgrImage(const StApi::IStImage &image, BgrFrame &frame) {
        const int width = static_cast<int>(image.GetImageWidth());
        const int height = static_cast<int>(image.GetImageHeight());
        const std::size_t row_bytes = static_cast<std::size_t>(width) * 3U;
        const auto *source = static_cast<const std::uint8_t *>(image.GetImageBuffer());
        const std::size_t source_pitch = image.GetImageLinePitch();

        frame.width = width;
        frame.height = height;
        frame.pixels.resize(row_bytes * static_cast<std::size_t>(height));
        for (int row = 0; row < height; ++row) {
            std::memcpy(frame.pixels.data() + static_cast<std::size_t>(row) * row_bytes,
                        source + static_cast<std::size_t>(row) * source_pitch, row_bytes);
        }
    }

    void captureLatestFrames(std::size_t index) noexcept {
        try {
            std::array<std::shared_ptr<BgrFrame>, kLatestFramePoolSize> frame_pool;
            for (auto &frame : frame_pool)
                frame = std::make_shared<BgrFrame>();

            while (running_.load() && !worker_failed_.load()) {
                StApi::IStStreamBufferReleasable *raw_buffer =
                    streams_[index]->RetrieveBuffer(10, StApi::StTimeoutHandling_Return);
                if (raw_buffer == nullptr)
                    continue;

                StApi::CIStStreamBufferPtr buffer(raw_buffer);
                if (!buffer->GetIStStreamBufferInfo()->IsImagePresent())
                    continue;

                const auto available = std::find_if(
                    frame_pool.begin(), frame_pool.end(),
                    [](const std::shared_ptr<BgrFrame> &frame) { return frame.use_count() == 1; });
                if (available == frame_pool.end())
                    continue;

                converters_[index]->Convert(buffer->GetIStImage(), converted_images_[index]);
                copyBgrImage(*converted_images_[index]->GetIStImage(), **available);
                const auto *buffer_info = buffer->GetIStStreamBufferInfo();
                (*available)->frame_id = buffer_info->GetFrameID();
                (*available)->timestamp_ns = buffer_info->GetTimestampNS();
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    latest_frames_[index] = *available;
                }
            }
        } catch (const GenICam::GenericException &error) {
            if (running_.load())
                recordWorkerError(error.GetDescription());
        } catch (const std::exception &error) {
            if (running_.load())
                recordWorkerError(error.what());
        } catch (...) {
            if (running_.load())
                recordWorkerError("unknown capture-thread error");
        }
    }

    void recordWorkerError(const std::string &error) {
        bool expected = false;
        if (!worker_failed_.compare_exchange_strong(expected, true))
            return;
        std::lock_guard<std::mutex> lock(error_mutex_);
        worker_error_ = error;
    }

    std::string workerError() const {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return worker_error_;
    }

    const BgrFrame &frameForIndex(std::size_t index) const {
        const std::shared_ptr<const BgrFrame> &frame = display_frames_.at(index);
        return frame ? *frame : empty_frame_;
    }

    void stopNoThrow() noexcept {
        running_.store(false);
        for (auto &device : devices_) {
            if (device.IsValid()) {
                try {
                    device->AcquisitionStop();
                } catch (...) {
                }
            }
        }
        for (auto &stream : streams_) {
            if (stream.IsValid()) {
                try {
                    stream->StopAcquisition();
                } catch (...) {
                }
            }
        }
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                try {
                    worker.join();
                } catch (...) {
                }
            }
        }
        for (auto &stream : streams_)
            stream.Reset();
        for (auto &converter : converters_)
            converter.Reset();
        for (auto &image : converted_images_)
            image.Reset();
        for (auto &device : devices_)
            device.Reset();
        system_.Reset();
    }

    StApi::CIStSystemPtr system_;
    std::array<StApi::CIStDevicePtr, kCameraCount> devices_;
    std::array<StApi::CIStDataStreamPtr, kCameraCount> streams_;
    std::array<StApi::CIStPixelFormatConverterPtr, kCameraCount> converters_;
    std::array<StApi::CIStImageBufferPtr, kCameraCount> converted_images_;
    std::array<std::thread, kCameraCount> workers_;
    std::array<std::shared_ptr<const BgrFrame>, kCameraCount> latest_frames_;
    std::array<std::shared_ptr<const BgrFrame>, kCameraCount> display_frames_;
    mutable std::mutex frame_mutex_;
    mutable std::mutex error_mutex_;
    std::atomic_bool running_ = false;
    std::atomic_bool worker_failed_ = false;
    std::string worker_error_;
    BgrFrame empty_frame_;
    std::string status_ = "Stopped";
};


void configureGenTlPath() {
    if (std::getenv("GENICAM_GENTL64_PATH") == nullptr &&
        setenv("GENICAM_GENTL64_PATH", FFS_SENTECH_GENTL_DIRECTORY, 0) != 0) {
        throw std::runtime_error("Unable to configure GENICAM_GENTL64_PATH");
    }
}

} // namespace

struct SentechStereoSource::Impl {
    std::optional<StApi::CStApiAutoInit> stapi;
    StereoCapture capture;
};

SentechStereoSource::SentechStereoSource() : impl_(std::make_unique<Impl>()) {}

SentechStereoSource::~SentechStereoSource() = default;

void SentechStereoSource::start() {
    try {
        configureGenTlPath();
        if (!impl_->stapi)
            impl_->stapi.emplace();
        impl_->capture.start();
    } catch (const GenICam::GenericException &error) {
        throw std::runtime_error(error.GetDescription());
    }
}

void SentechStereoSource::stop() {
    impl_->capture.stop();
}

void SentechStereoSource::poll() {
    impl_->capture.poll();
}

bool SentechStereoSource::running() const noexcept {
    return impl_->capture.running();
}

const std::string &SentechStereoSource::status() const {
    return impl_->capture.status();
}

const BgrFrame &SentechStereoSource::leftFrame() const {
    return impl_->capture.leftFrame();
}

const BgrFrame &SentechStereoSource::rightFrame() const {
    return impl_->capture.rightFrame();
}

} // namespace ffs_viewer::io
