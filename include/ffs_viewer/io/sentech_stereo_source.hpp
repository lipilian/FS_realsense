#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ffs_viewer::io {

struct BgrFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    bool valid() const {
        return width > 0 && height > 0 && !pixels.empty();
    }
};

class SentechStereoSource final {
  public:
    SentechStereoSource();
    ~SentechStereoSource();

    SentechStereoSource(const SentechStereoSource &) = delete;
    SentechStereoSource &operator=(const SentechStereoSource &) = delete;

    // Starts the two configured cameras and their latest-frame capture workers.
    void start();
    void stop();
    void poll();

    bool running() const noexcept;
    const std::string &status() const;
    const BgrFrame &leftFrame() const;
    const BgrFrame &rightFrame() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ffs_viewer::io
