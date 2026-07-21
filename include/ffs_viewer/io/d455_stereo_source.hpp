#pragma once

#include "ffs_viewer/io/stereo_source.hpp"

#include <cstdint>
#include <memory>

namespace ffs_viewer::io {

class D455StereoSource final : public StereoSource {
public:
    D455StereoSource();
    ~D455StereoSource() override;

    D455StereoSource(const D455StereoSource&) = delete;
    D455StereoSource& operator=(const D455StereoSource&) = delete;

    // Starts IR1 and IR2 at the fixed FFS engine profile: 1280x800 Y8 @ 15 Hz.
    void open();
    bool isOpen() const noexcept;

    // Waits for a pair, drains queued pairs, and returns only the newest one.
    bool next(StereoFrame& frame) override;
    StereoCalibration calibration() const override;
    std::uint64_t droppedPairs() const noexcept;

private:
    struct Impl;

    StereoCalibration calibration_{};
    std::unique_ptr<Impl> impl_;
};

}  // namespace ffs_viewer::io
