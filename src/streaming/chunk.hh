#pragma once

#include "compression.params.hh"

#include <mutex>
#include <optional>
#include <vector>

namespace zarr {
class Chunk
{
  public:
    Chunk(size_t size_bytes, size_t bytes_per_px);
    ~Chunk() = default;

    void write_tile(const std::vector<uint8_t>& tile);

    bool has_data() const;
    size_t size_bytes() const;
    std::vector<uint8_t>&& compress_and_take_buffer(
      const std::optional<CompressionParams>& compression_params);

  private:
    const size_t size_bytes_;
    const size_t bytes_per_px_;

    std::mutex mutex_;
    std::vector<uint8_t> buffer_;
    size_t offset_;

    bool has_data_;

    void reset_();

    bool compress_(const BloscCompressionParams& params);
    bool compress_(const ZstdCompressionParams& params);
};
} // namespace zarr