#include "file.sink.hh"
#include "macros.hh"
#include "s3.sink.hh"
#include "shard.hh"

zarr::Shard::Shard(ShardConfig&& config,
                   std::shared_ptr<ThreadPool> thread_pool)
  : config_(std::move(config))
  , thread_pool_(thread_pool)
  , frames_written_(0)
  , file_offset_(0)
{
    EXPECT(config_.dims != nullptr, "ArrayDimensions pointer cannot be null");
    EXPECT(thread_pool_ != nullptr, "ThreadPool pointer cannot be null");

    chunk_indices_ = config_.dims->chunk_indices_for_shard(config_.shard_index);

    const size_t n_chunks = chunk_indices_.size();
    chunks_.resize(n_chunks);
    offsets_.resize(n_chunks);
    extents_.resize(n_chunks);

    frames_per_layer_ = config_.dims->at(0).chunk_size_px;
    for (auto i = 1; i < config_.dims->ndims() - 2; ++i) {
        frames_per_layer_ *= config_.dims->at(i).array_size_px;
    }
    EXPECT(frames_per_layer_ > 0, "Frames per layer computed to be 0");

    layers_per_shard_ = config_.dims->at(0).shard_size_chunks;
}

size_t
zarr::Shard::write_frame(const std::vector<uint8_t>& frame)
{
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

    const auto frame_id = frames_written_;

    // offset among the chunks in the lattice
    const uint32_t group_offset = dims->tile_group_offset(frame_id);

    // offset within each chunk
    const size_t chunk_offset = dims->chunk_internal_offset(frame_id);

    const size_t group_high = group_offset + dims->number_of_chunks_in_memory();

    const uint8_t* frame_ptr = frame.data();
    const size_t frame_size = frame.size();

    size_t bytes_written = 0;

    for (auto i = 0; i < chunk_indices_.size(); ++i) {
        const auto chunk_idx = chunk_indices_[i];
        if (chunk_idx < group_offset || chunk_idx >= group_high) {
            continue;
        }
        auto& chunk = chunks_[i];
        {
            const auto chunk_size = chunk.size();
            EXPECT(chunk_size == bytes_per_chunk,
                   "Expected chunk to have size ",
                   bytes_per_chunk,
                   ", got ",
                   chunk_size);
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
                bytes_written += nbytes;
            }
            chunk_pos += bytes_per_row;
        }
    }

    if (++frames_written_ % frames_per_layer_ == 0) {
        EXPECT(close_current_layer_(), "Failed to close current layer");
    }
    return bytes_written;
}

bool
zarr::Shard::close()
{
    if (!flush_chunks_()) {
        LOG_ERROR("Failed to flush chunks.");
        return false;
    }

    return true;
}

bool
zarr::Shard::close_current_layer_()
{
    if (!flush_chunks_()) {
        LOG_ERROR("Failed to flush chunks");
        return false;
    }

    current_layer_ = (current_layer_ + 1) % layers_per_shard_;
    active_chunks_.clear();

    return true;
}
