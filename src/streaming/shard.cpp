#include "file.sink.hh"
#include "macros.hh"
#include "s3.sink.hh"
#include "shard.hh"

#include <blosc.h>

#include <ranges>

zarr::Shard::Shard(ShardConfig&& config,
                   std::shared_ptr<ThreadPool> thread_pool)
  : config_(std::move(config))
  , thread_pool_(thread_pool)
  , file_offset_(0)
  , current_layer_(0)
{
    EXPECT(config_.dims != nullptr, "ArrayDimensions pointer cannot be null");
    EXPECT(thread_pool_ != nullptr, "ThreadPool pointer cannot be null");

    const auto& chunk_indices =
      config_.dims->chunk_indices_for_shard(config_.shard_grid_index);

    for (const auto& chunk_idx : chunk_indices) {
        chunks_.emplace(chunk_idx, std::vector<uint8_t>());
    }

    layers_per_shard_ = config_.dims->at(0).shard_size_chunks;
    EXPECT(layers_per_shard_ > 0,
           "Shard size in append dimension cannot be zero");

    const size_t n_chunks_this_shard = chunk_indices.size();
    offsets_.resize(n_chunks_this_shard);
    std::ranges::fill(offsets_, std::numeric_limits<uint64_t>::max());

    extents_.resize(n_chunks_this_shard);
    std::ranges::fill(extents_, std::numeric_limits<uint64_t>::max());

    chunks_per_layer_ =
      (n_chunks_this_shard + layers_per_shard_ - 1) / layers_per_shard_;

    frames_per_layer_ = config_.dims->at(0).chunk_size_px;
    for (auto i = 1; i < config_.dims->ndims() - 2; ++i) {
        frames_per_layer_ *= config_.dims->at(i).array_size_px;
    }
    EXPECT(frames_per_layer_ > 0, "Frames per layer computed to be 0");

    const uint64_t frames_per_shard = frames_per_layer_ * layers_per_shard_;
    frame_lower_bound_ = config_.append_shard_index * frames_per_shard +
                         frames_per_layer_ * current_layer_;
    frame_upper_bound_ = frame_lower_bound_ + frames_per_layer_;

    for (auto layer = 0; layer < layers_per_shard_; ++layer) {
        bytes_to_flush_.emplace(layer, 0);
        mutexes_.emplace(layer, std::make_unique<std::mutex>());
    }
}

zarr::Shard::~Shard()
{
    for (auto& future : compress_futures_ | std::views::values) {
        future.wait();
    }
    compress_promises_.clear();
    compress_futures_.clear();

    for (auto& future : flush_futures_ | std::views::values) {
        future.wait();
    }
    flush_promises_.clear();
    flush_futures_.clear();
}

size_t
zarr::Shard::write_frame(const std::span<uint8_t>& frame, uint64_t frame_id)
{
    std::unique_lock lock(*mutexes_[current_layer_]);

    // check that this frame belongs in this layer
    assert_frame_in_layer_(frame_id);

    const auto& dims = config_.dims;
    const size_t bytes_per_px = dims->bytes_of_type();

    const size_t frame_cols = dims->width_dim().array_size_px;
    const size_t tile_cols = dims->width_dim().chunk_size_px;

    const size_t frame_rows = dims->height_dim().array_size_px;
    const size_t tile_rows = dims->height_dim().chunk_size_px;

    const size_t bytes_per_chunk = dims->bytes_per_chunk();

    if (tile_cols == 0 || tile_rows == 0) {
        return 0;
    }

    const size_t bytes_per_row = tile_cols * bytes_per_px;
    const size_t n_tiles_x = (frame_cols + tile_cols - 1) / tile_cols;
    if (!dims->frame_is_in_shard(frame_id, config_.shard_grid_index)) {
        if ((frame_id + 1) % frames_per_layer_ == 0) {
            EXPECT(close_current_layer_(), "Failed to close current layer");
        }
        return 0;
    }

    // offset among the chunks in the lattice
    const uint32_t group_offset = dims->tile_group_offset(frame_id);
    std::vector<uint32_t> chunk_indices = dims->chunk_indices_for_shard_layer(
      config_.shard_grid_index, current_layer_);

    // offset within each chunk
    const size_t chunk_offset = dims->chunk_internal_offset(frame_id);

    const uint8_t* frame_ptr = frame.data();
    const size_t frame_size = frame.size();

    size_t chunks_written_to = 0;
    size_t bytes_written = 0;

    for (auto& chunk_idx : chunk_indices) {
        // wait for this chunk buffer to come available
        if (const auto it = flush_futures_.find(chunk_idx);
            it != flush_futures_.end()) {
            auto& future = it->second;
            future.wait();
            if (!future.get()) {
                LOG_ERROR("Prior flush to chunk ",
                          chunk_idx,
                          " failed. Aborting write");
                return -1;
            }

            flush_promises_.erase(chunk_idx);
            flush_futures_.erase(chunk_idx);
        }

        size_t bytes_written_this_chunk = 0;
        auto& chunk = chunks_[chunk_idx];
        if (chunk.size() != bytes_per_chunk) {
            chunk.resize(bytes_per_chunk);
            std::ranges::fill(chunk, 0);
        }

        const size_t tile_idx = chunk_idx - group_offset;
        const size_t tile_idx_y = tile_idx / n_tiles_x;
        const size_t tile_idx_x = tile_idx % n_tiles_x;

        uint8_t* chunk_start = chunk.data();
        size_t chunk_pos = chunk_offset;

        for (auto k = 0; k < tile_rows; ++k) {
            const auto frame_row = tile_idx_y * tile_rows + k;
            if (frame_row < frame_rows) {
                const auto frame_col = tile_idx_x * tile_cols;

                const auto region_width =
                  std::min(frame_col + tile_cols, frame_cols) - frame_col;

                const auto region_start =
                  bytes_per_px * (frame_row * frame_cols + frame_col);
                const auto nbytes = region_width * bytes_per_px;

                // copy region
                EXPECT(region_start + nbytes <= frame_size,
                       "Buffer overflow in framme. Region start: ",
                       region_start,
                       " nbytes: ",
                       nbytes,
                       " data size: ",
                       frame_size);
                EXPECT(chunk_pos + nbytes <= bytes_per_chunk,
                       "Buffer overflow in chunk. Chunk pos: ",
                       chunk_pos,
                       " nbytes: ",
                       nbytes,
                       " bytes per chunk: ",
                       bytes_per_chunk);
                memcpy(
                  chunk_start + chunk_pos, frame_ptr + region_start, nbytes);
                bytes_written_this_chunk += nbytes;
            }
            chunk_pos += bytes_per_row;
        }

        ++chunks_written_to;
        bytes_written += bytes_written_this_chunk;
    }

    bytes_to_flush_[current_layer_] += bytes_written;

    if ((frame_id + 1) % frames_per_layer_ == 0) {
        EXPECT(close_current_layer_(), "Failed to close current layer");
    }
    return bytes_written;
}

bool
zarr::Shard::close()
{
    for (auto [layer, bytes_to_flush] : bytes_to_flush_) {
        if (bytes_to_flush > 0) {
            if (!flush_chunks_(layer)) {
                LOG_ERROR("Failed to flush chunks to layer ", layer);
                return false;
            }
        }
    }

    bool success = true;

    for (auto& future : flush_futures_ | std::views::values) {
        future.wait();
        success &= future.get();
    }
    flush_promises_.clear();
    flush_futures_.clear();

    if (success) {
        clean_up_resource_();
    }

    return success;
}

void
zarr::Shard::assert_frame_in_layer_(uint64_t frame_id) const
{
    EXPECT(frame_id >= frame_lower_bound_ && frame_id < frame_upper_bound_,
           "Frame ",
           frame_id,
           " is not in the current shard layer (minimum ",
           frame_lower_bound_,
           ", maximum ",
           frame_upper_bound_ - 1,
           ").");
}

bool
zarr::Shard::close_current_layer_()
{
    if (!compress_and_flush_data_(current_layer_)) {
        LOG_ERROR("Failed to flush chunks");
        return false;
    }

    current_layer_ = (current_layer_ + 1) % layers_per_shard_;

    // update upper and lower bounds
    const uint64_t frames_per_shard = frames_per_layer_ * layers_per_shard_;
    frame_lower_bound_ = config_.append_shard_index * frames_per_shard +
                         frames_per_layer_ * current_layer_;
    frame_upper_bound_ = frame_lower_bound_ + frames_per_layer_;

    return true;
}

bool
zarr::Shard::compress_and_flush_data_(uint32_t layer)
{
    return compress_chunks_(layer) && flush_chunks_(layer);
}

bool
zarr::Shard::compress_chunks_(uint32_t layer)
{
    std::unique_lock lock(*mutexes_[layer]);

    const auto chunk_indices = config_.dims->chunk_indices_for_shard_layer(
      config_.shard_grid_index, layer);

    if (!config_.compression_params) {
        for (const auto& idx : chunk_indices) {
            compress_promises_.emplace(idx, std::promise<bool>());
            auto& promise = compress_promises_[idx];

            compress_futures_.emplace(idx, promise.get_future());
            promise.set_value(true);
        }

        return true;
    }

    auto job = [this, chunk_indices](std::string& err) {
        bool success = true;
        const auto& params = config_.compression_params;
        const size_t bytes_per_px = config_.dims->bytes_of_type();

        size_t offset = file_offset_;

        for (const auto& index : chunk_indices) {
            std::vector<uint8_t>& chunk = chunks_[index];
            const size_t bytes_of_chunk = chunk.size();
            const uint32_t internal_index =
              config_.dims->shard_internal_index(index);

            compress_promises_.emplace(index, std::promise<bool>());
            compress_futures_.emplace(index,
                                      compress_promises_[index].get_future());

            try {
                int nbytes_compressed = bytes_of_chunk + BLOSC_MAX_OVERHEAD;
                std::vector<uint8_t> compressed(nbytes_compressed);

                nbytes_compressed = blosc_compress_ctx(params->clevel,
                                                       params->shuffle,
                                                       bytes_per_px,
                                                       chunk.size(),
                                                       chunk.data(),
                                                       compressed.data(),
                                                       compressed.size(),
                                                       params->codec_id.c_str(),
                                                       0,
                                                       1);
                if (nbytes_compressed <= 0) {
                    err = "blosc_compress_ctx failed with code " +
                          std::to_string(nbytes_compressed) + " for chunk " +
                          std::to_string(internal_index) + " of shard ";
                    success = false;
                    compress_promises_.at(index).set_value(false);
                } else {
                    offsets_[internal_index] = offset;
                    extents_[internal_index] = compressed.size();
                    chunk.swap(compressed);

                    offset += nbytes_compressed;
                    compress_promises_.at(index).set_value(true);
                }
            } catch (const std::exception& exc) {
                err = "Failed to compress: " + std::string(exc.what());
                success = false;
                compress_promises_.at(index).set_value(false);
            }
        }

        return success;
    };

    if (thread_pool_->n_threads() == 1 || !thread_pool_->push_job(job)) {
        if (std::string err; !job(err)) {
            LOG_ERROR(err);
            return false;
        }
    }

    return true;
}

bool
zarr::Shard::flush_chunks_(uint32_t layer)
{
    std::unique_lock lock(*mutexes_[layer]);

    const auto chunk_indices = config_.dims->chunk_indices_for_shard_layer(
      config_.shard_grid_index, layer);
    const size_t bytes_per_chunk = config_.dims->bytes_per_chunk();

    bool flush_success = true;
    for (const auto& chunk_idx : chunk_indices) {
        auto& chunk = chunks_[chunk_idx];

        flush_promises_.emplace(chunk_idx, std::promise<bool>());
        flush_futures_.emplace(chunk_idx,
                               flush_promises_[chunk_idx].get_future());
        auto job =
          [this, bytes_per_chunk, chunk_idx, layer, &chunk](std::string& err) {
              bool success;

              try {
                  compress_futures_[chunk_idx].wait();
                  const bool compressed_successfully =
                    compress_futures_[chunk_idx].get();

                  compress_promises_.erase(chunk_idx);
                  compress_futures_.erase(chunk_idx);

                  // failed to compress
                  if (compressed_successfully) {
                      success = write_to_offset_(chunk, offsets_[chunk_idx]);

                      if (success) {
                          bytes_to_flush_[layer] = 0;

                          // free up memory until next time we get to this layer
                          if (layers_per_shard_ > 0) {
                              chunk.clear();
                          } else {
                              chunk.resize(bytes_per_chunk);
                              std::ranges::fill(chunk, 0);
                          }
                      }
                  } else {
                      err =
                        "Failed to flush chunk: " + std::to_string(chunk_idx) +
                        ": compression failed";
                      success = false;
                  }
              } catch (const std::exception& exc) {
                  err = "Failed to flush chunk " + std::to_string(chunk_idx) +
                        ": " + std::string(exc.what());
                  success = false;
              }

              flush_promises_[chunk_idx].set_value(success);
              return success;
          };

        if (thread_pool_->n_threads() == 1 || !thread_pool_->push_job(job)) {
            if (std::string err; !job(err)) {
                LOG_ERROR(err);
                flush_success = false;
            }
        }
    }

    return flush_success;
}