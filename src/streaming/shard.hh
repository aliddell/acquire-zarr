#pragma once

#include "array.dimensions.hh"
#include "locked.buffer.hh"
#include "thread.pool.hh"

#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

namespace zarr {
struct ShardConfig
{
    uint32_t shard_index;
    std::shared_ptr<ArrayDimensions> dims;
    std::string path;
};

class Shard
{
  public:
    Shard(ShardConfig&& config, std::shared_ptr<ThreadPool> thread_pool);
    virtual ~Shard() = default;

    /**
     * @brief Write a frame to this shard.
     * @param frame The frame to write.
     * @param frame_id The ID of the frame to write.
     * @return The number of bytes written.
     */
    [[nodiscard]] size_t write_frame(const std::span<uint8_t>& frame,
                                     uint64_t frame_id);

    /**
     * @brief Close the shard file.
     * @return True if successfully closed, otherwise false.
     */
    [[nodiscard]] bool close();

  protected:
    ShardConfig config_;
    std::shared_ptr<ThreadPool> thread_pool_;

    uint64_t frames_per_layer_;
    uint64_t chunks_per_layer_;
    uint64_t layers_per_shard_;

    std::unordered_map<uint32_t, std::vector<uint8_t>> chunks_;

    std::vector<uint64_t> offsets_;
    std::vector<uint64_t> extents_;

    uint64_t file_offset_;
    uint32_t current_layer_;

    /**
     * @brief Close the current layer, flush chunks, write the table, and
     * increment the current layer.
     * @return
     */
    [[nodiscard]] bool close_current_layer_();

    /**
     * @brief
     * @return
     */
    [[nodiscard]] virtual bool compress_and_flush_chunks_() = 0;
};
} // namespace zarr