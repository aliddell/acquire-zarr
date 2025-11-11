#include "shard.hh"
#include "unit.test.macros.hh"
#include "zarr.common.hh"

namespace {
class TestShard : public zarr::Shard
{
  public:
    explicit TestShard(zarr::ShardConfig&& config,
                       std::shared_ptr<zarr::ThreadPool> thread_pool)
      : Shard(std::move(config), thread_pool)
    {
    }

    const std::unordered_map<uint32_t, std::vector<uint8_t>>& chunks()
    {
        return chunks_;
    }
    size_t flush_count() const { return flush_count_; }

  protected:
    bool flush_chunks_() override
    {
        ++flush_count_;
        return true;
    }

  private:
    size_t flush_count_{ 0 };
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

    size_t frames_per_chunk = dims[0].chunk_size_px;
    for (auto i = 1; i < dims.size() - 2; ++i) {
        frames_per_chunk *= dims[i].array_size_px;
    }

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
            .shard_index = 0,
            .dims = array_dimensions,
            .path = TEST ".zarr",
        };

        TestShard shard(std::move(config), thread_pool);
        CHECK(shard.flush_count() == 0);

        std::vector<uint16_t> frame(2048 * 2048, 1);
        const std::span frame_span{ reinterpret_cast<uint8_t*>(frame.data()),
                                    frame.size() * sizeof(uint16_t) };

        for (auto i = 0; i < frames_per_chunk - 1; ++i) {
            const size_t expected_bytes_written =
              array_dimensions->frame_in_shard(i, 0) ? frame_span.size() / 4
                                                     : 0;
            const size_t bytes_written = shard.write_frame(frame_span, i);

            // we have 4 shards, so we should only have written 1/4 of the frame
            EXPECT(bytes_written == expected_bytes_written,
                   "Expected to write most ",
                   expected_bytes_written,
                   " bytes, wrote ",
                   bytes_written,
                   " (frame ",
                   i,
                   ")");

            // should not have flushed
            CHECK(shard.flush_count() == 0);
        }

        // write the final frame in the chunk
        {
            const size_t bytes_written =
              shard.write_frame(frame_span, frames_per_chunk - 1);
            const size_t expected_bytes_written =
              array_dimensions->frame_in_shard(frames_per_chunk - 1, 0)
                ? frame_span.size() / 4
                : 0;

            // we have 4 shards, so we should only have written 1/4 of the frame
            EXPECT(bytes_written == expected_bytes_written,
                   "Expected to write ",
                   expected_bytes_written,
                   " bytes, wrote ",
                   bytes_written);

            // should have flushed
            CHECK(shard.flush_count() == 1);
        }

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR(e.what());
    }

    return retval;
}