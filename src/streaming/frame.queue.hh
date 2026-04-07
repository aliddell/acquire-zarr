#pragma once

#include "definitions.hh"
#include "locked.buffer.hh"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>

namespace zarr {
class FrameQueue
{
  public:
    explicit FrameQueue(size_t num_frames, size_t avg_frame_size);
    ~FrameQueue() = default;

    bool push(LockedBuffer& frame,
              const std::string& key,
              const std::optional<uint64_t>& frame_id,
              const std::optional<uint64_t>& timestamp);
    bool pop(LockedBuffer& frame,
             std::string& key,
             std::optional<uint64_t>& frame_id,
             std::optional<uint64_t>& timestamp);

    size_t size() const;
    size_t bytes_used() const;
    bool full() const;
    bool empty() const;
    void clear();

  private:
    struct Frame
    {
        std::string key;
        LockedBuffer data;
        std::optional<uint64_t> frame_id;
        std::optional<uint64_t> timestamp;
        std::atomic<bool> ready{ false };
    };

    std::vector<Frame> buffer_;
    size_t capacity_;

    // Producer and consumer positions
    std::atomic<size_t> write_pos_{ 0 };
    std::atomic<size_t> read_pos_{ 0 };

    std::mutex mutex_;
};
} // namespace zarr