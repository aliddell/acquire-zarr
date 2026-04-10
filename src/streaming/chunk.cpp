#include "chunk.hh"
#include "macros.hh"
#include "zarr.common.hh"

#include <algorithm>

zarr::Chunk::Chunk(size_t size_bytes, size_t bytes_per_px)
  : size_bytes_(size_bytes)
  , bytes_per_px_(bytes_per_px)
  , offset_(0)
  , has_data_(false)
{
    EXPECT(size_bytes_ > 0, "Empty chunk");

    reset_();
}

void
zarr::Chunk::write_tile(const std::vector<uint8_t>& tile)
{
    if (!has_data_) {
        bool any_nonzero = false;

        // check to see if any byte is nonzero
        for (const auto& byte : tile) {
            if (byte != 0) {
                any_nonzero = true;
                break;
            }
        }

        if (any_nonzero) {
            memcpy(buffer_.data() + offset_, tile.data(), tile.size());
            has_data_ = true;
        }

        offset_ += tile.size();
        return;
    }

    memcpy(buffer_.data() + offset_, tile.data(), tile.size());
    offset_ += tile.size();
}

bool
zarr::Chunk::has_data() const
{
    return has_data_;
}

size_t
zarr::Chunk::size_bytes() const
{
    return size_bytes_;
}

std::vector<uint8_t>&&
zarr::Chunk::compress_and_take_buffer(
  const std::optional<CompressionParams>& compression_params)
{
    if (compression_params) {
        bool compressed = std::visit(
          [this]<typename ParamT>(const ParamT& params) {
              using T = std::decay_t<ParamT>;
              if constexpr (std::is_same_v<T, BloscCompressionParams>) {
                  return compress_(params);
              } else {
                  return compress_(params);
              }
          },
          *compression_params);
        EXPECT(compressed, "Failed to compress chunk buffer");
    }

    auto buffer = std::move(buffer_);
    reset_();

    return std::move(buffer);
}

void
zarr::Chunk::reset_()
{
    offset_ = 0;
    buffer_.resize(size_bytes_);
    std::ranges::fill(buffer_, 0);
}

bool
zarr::Chunk::compress_(const BloscCompressionParams& params)
{
    std::unique_lock lock(mutex_);
    return compress_in_place(buffer_, params, bytes_per_px_);
}

bool
zarr::Chunk::compress_(const ZstdCompressionParams& params)
{
    std::unique_lock lock(mutex_);
    return compress_in_place(buffer_, params);
}