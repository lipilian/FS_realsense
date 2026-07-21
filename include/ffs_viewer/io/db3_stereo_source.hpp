#pragma once

#include "ffs_viewer/io/stereo_source.hpp"

#include <memory>
#include <string>

namespace ffs_viewer::io {

class Db3StereoSource final : public StereoSource {
public:
    explicit Db3StereoSource(std::string recording_path); // explicit constructor to avoid implicit conversions
    ~Db3StereoSource() override; // override destructor to ensure proper cleanup of derived class

    Db3StereoSource(const Db3StereoSource&) = delete; // no copy
    Db3StereoSource& operator=(const Db3StereoSource&) = delete; // no copy

    void open(); // open db3 pipeline, enable calibration() and next()
    bool isOpen() const noexcept; // check if db3 pipeline is open, return true if open, false otherwise

    bool next(StereoFrame& frame) override; // override next() to retrieve the next stereo frame from the db3 pipeline
    StereoCalibration calibration() const override; // override calibration() to retrieve the stereo calibration parameters from the db3 pipeline

private:
    struct Impl; // Pointer to implementation. Private object, will specify in .cpp. 

    std::string recording_path_; // path to db3 file.
    StereoCalibration calibration_{}; // stereo calibration parameters, 0 initialized. Will be set in open() function.
    std::unique_ptr<Impl> impl_; // unique pointer to implementation. Will be initialized in constructor, and deleted in destructor.
};

}  // namespace ffs_viewer::io
