#include "shard.hh"
#include "unit-test-utils.hh"
#include "zarr.common.hh"

#include <ranges>

const static std::string shard_file_path(TEST ".bin");

namespace {
class TestShard final : public zarr::Shard
{
  public:
    explicit TestShard(zarr::ShardConfig&& config,
                       std::shared_ptr<zarr::ThreadPool> thread_pool)
      : Shard(std::move(config), thread_pool)
    {
    }

    const std::map<uint32_t, std::vector<uint8_t>>& chunks() { return chunks_; }
    bool has_flushed() const { return has_flushed_; }

  protected:
    // this is called when chunks are flushed
    bool write_to_offset_(const std::vector<uint8_t>& chunk,
                          size_t offset) override
    {
        has_flushed_.store(true);
        return true;
    }

    void clean_up_resource_() override {}

  private:
    std::atomic<bool> has_flushed_{ false };
};
} // namespace

int
main()
{
    int retval = 1;

    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      0, [](const std::string& err) { LOG_ERROR(err); });

    std::vector<ZarrDimension> dims(5);
    dims[0] = ZarrDimension("t", ZarrDimensionType_Time, 0, 32, 2);
    dims[1] = ZarrDimension("c", ZarrDimensionType_Channel, 3, 1, 1);
    dims[2] = ZarrDimension("z", ZarrDimensionType_Space, 128, 128, 1);
    dims[3] = ZarrDimension("y", ZarrDimensionType_Space, 2048, 128, 8);
    dims[4] = ZarrDimension("x", ZarrDimensionType_Space, 2048, 128, 8);

    size_t chunks_per_layer = 1;
    for (auto i = 1; i < dims.size(); ++i) {
        chunks_per_layer *= dims[i].shard_size_chunks;
    }
    const size_t chunks_per_shard =
      chunks_per_layer * dims[0].shard_size_chunks;

    try {
        auto array_dimensions = std::make_shared<ArrayDimensions>(
          std::move(dims), ZarrDataType_uint16);
        const size_t bytes_per_chunk = array_dimensions->bytes_per_chunk();

        const size_t tile_size_px =
          array_dimensions->width_dim().chunk_size_px *
          array_dimensions->height_dim().chunk_size_px;
        const size_t tile_size_bytes =
          tile_size_px * array_dimensions->bytes_of_type();

        zarr::ShardConfig config{
            .shard_grid_index = 0,
            .append_shard_index = 0,
            .dims = array_dimensions,
            .path = shard_file_path,
        };

        TestShard shard(std::move(config), thread_pool);

        CHECK(!shard.has_flushed());

        const auto& chunks = shard.chunks();
        EXPECT(chunks.size() == chunks_per_shard,
               "Expected ",
               chunks_per_shard,
               " active chunks in shard, got ",
               chunks.size());

        std::vector<uint16_t> frame(2048 * 2048, 1);
        const std::span frame_span{ reinterpret_cast<uint8_t*>(frame.data()),
                                    frame.size() * sizeof(uint16_t) };
        const size_t bytes_written = shard.write_frame(frame_span, 0);

        // we have 4 shards, so we should only have written 1/4 of the frame
        EXPECT(bytes_written == frame_span.size() / 4,
               "Expected to write ",
               frame_span.size() / 4,
               " bytes, wrote ",
               bytes_written);

        const auto chunk_indices_first_layer =
          array_dimensions->chunk_indices_for_shard_layer(0, 0);

        // check that each tile has been written correctly
        for (auto chunk_idx : chunk_indices_first_layer) {
            const auto& chunk = chunks.at(chunk_idx);
            EXPECT(chunk.size() == bytes_per_chunk,
                   "Expected chunk at index ",
                   chunk_idx,
                   " to have size ",
                   bytes_per_chunk,
                   " got ",
                   chunk.size());
            std::span tile = { reinterpret_cast<const uint16_t*>(chunk.data()),
                               tile_size_px };
            for (const auto& px : tile) {
                EXPECT(px == 1, "Unexpected pixel value: ", px);
            }

            // other pixel values should be unaffected
            for (auto i = tile_size_bytes; i < chunk.size(); ++i) {
                EXPECT(chunk[i] == 0, "Unexpected byte in chunk: ", chunk[i]);
            }
        }

        // chunks in second layer should be empty
        const auto chunk_indices_second_layer =
          array_dimensions->chunk_indices_for_shard_layer(0, 1);
        for (const auto& chunk_idx : chunk_indices_second_layer) {
            const auto& chunk = chunks.at(chunk_idx);
            EXPECT(chunk.empty(),
                   "Expected chunk ",
                   chunk_idx,
                   " to be empty, got size ",
                   chunk.size());
        }

        // should not have flushed
        CHECK(!shard.has_flushed());

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR(e.what());
    }

    return retval;
}