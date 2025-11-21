#pragma once

#include "array.dimensions.hh"
#include "locked.buffer.hh"
#include "thread.pool.hh"

#include <future>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace zarr {
struct ShardConfig
{
    uint32_t shard_grid_index;
    uint32_t append_shard_index;
    std::shared_ptr<ArrayDimensions> dims;
    std::optional<BloscCompressionParams> compression_params;
    std::string path;
};

class Shard
{
  public:
    Shard(ShardConfig&& config, std::shared_ptr<ThreadPool> thread_pool);
    virtual ~Shard();

    /**
     * @brief Write a frame to this shard.
     * @param frame The frame to write.
     * @param frame_id The ID of the frame to write.
     * @throws std::runtime_error if the frame ID does not belong to this layer.
     * @throws std::runtime_error if the layer needs to be closed and fails to
     * do so.
     * @return The number of bytes written, if any, or -1 if an error was
     * encountered.
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

    // key: chunk internal index, value: chunk buffer
    std::map<uint32_t, std::vector<uint8_t>> chunks_;

    // key: chunk internal index, value: promise
    std::map<uint32_t, std::promise<bool>> compress_promises_;
    // key: chunk internal index, value: future
    std::map<uint32_t, std::future<bool>> compress_futures_;

    // key: chunk internal index, value: promise
    std::map<uint32_t, std::promise<bool>> flush_promises_;
    // key: chunk internal index, value: future
    std::map<uint32_t, std::future<bool>> flush_futures_;

    // key: layer, value: bytes to flush in a given layer
    std::map<uint32_t, uint64_t> bytes_to_flush_;

    // key: layer, value: mutex for the chunks in this layer
    std::map<uint32_t, std::unique_ptr<std::mutex>> mutexes_;

    std::vector<uint64_t> offsets_;
    std::vector<uint64_t> extents_;

    uint64_t file_offset_;
    uint32_t current_layer_;

    uint64_t frame_lower_bound_;
    uint64_t frame_upper_bound_;

    std::mutex mutex_;
    std::condition_variable chunk_cv_;

    /**
     * @brief Check that the frame @p frame_id is correct for this shard layer.
     * @param frame_id The ID of the frame.
     * @throws std::runtime_error if @p frame_id is not between the frame bounds
     * for this append shard index and the current layer.
     */
    void assert_frame_in_layer_(uint64_t frame_id) const;

    /**
     * @brief Close the current layer, flush chunks, write the table, and
     * increment the current layer.
     * @return True if the layer was closed successfully, otherwise false.
     */
    [[nodiscard]] bool close_current_layer_();

    /**
     * @brief Compress the chunks (if applicable) and flush them to disk or S3.
     * @return True if the current chunk layer was compressed and flushed.
     */
    [[nodiscard]] bool compress_and_flush_data_(uint32_t layer);

    /**
     * @brief Compress the chunks, if applicable.
     * @return True if not compressing, or if compression completes
     * successfully. False otherwise.
     */
    [[nodiscard]] bool compress_chunks_(uint32_t layer);

    /**
     * @brief Flush the compressed chunks to disk or S3.
     * @return True if the current chunk layer was flushed, otherwise false.
     */
    [[nodiscard]] bool flush_chunks_(uint32_t layer);

    /**
     * @brief Write a (possibly compressed) chunk to disk or S3 at the offset.
     * @param chunk The chunk data.
     * @param offset The offset to write to.
     * @return True if all bytes of @p chunk were flushed.
     */
    [[nodiscard]] virtual bool write_to_offset_(
      const std::vector<uint8_t>& chunk,
      size_t offset) = 0;

    /**
     * @brief Clean up the underlying storage resource.
     * @details Releases a file handle or finalizes an upload.
     */
    virtual void clean_up_resource_() = 0;
};
} // namespace zarr