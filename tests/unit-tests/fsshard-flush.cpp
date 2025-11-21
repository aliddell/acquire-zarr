#include "fs.shard.hh"
#include "unit.test.macros.hh"
#include "zarr.common.hh"

#include <filesystem>

namespace fs = std::filesystem;

const static std::string shard_file_path(TEST ".bin");

int
main()
{
    int retval = 1;

    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      1, [](const std::string& err) { std::cerr << err << std::endl; });

    auto file_handle_pool = std::make_shared<zarr::FileHandlePool>();

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

    size_t bytes_per_chunk = zarr::bytes_of_type(ZarrDataType_uint16);
    size_t chunks_per_shard = 1;
    for (const auto& dim : dims) {
        bytes_per_chunk *= dim.chunk_size_px;
        chunks_per_shard *= dim.shard_size_chunks;
    }

    // just flushing one layer
    const size_t expected_file_size =
      bytes_per_chunk * chunks_per_shard / dims[0].shard_size_chunks;

    try {
        auto array_dimensions = std::make_shared<ArrayDimensions>(
          std::move(dims), ZarrDataType_uint16);

        zarr::ShardConfig config{
            .shard_grid_index = 0,
            .append_shard_index = 0,
            .dims = array_dimensions,
            .path = shard_file_path,
        };

        zarr::FSShard shard(std::move(config), thread_pool, file_handle_pool);

        if (fs::exists(shard_file_path)) {
            fs::remove_all(shard_file_path);
        }

        std::vector<uint16_t> frame(2048 * 2048, 1);
        const std::span frame_span{ reinterpret_cast<uint8_t*>(frame.data()),
                                    frame.size() * sizeof(uint16_t) };

        for (auto i = 0; i < frames_per_chunk - 1; ++i) {
            const size_t expected_bytes_written =
              array_dimensions->frame_is_in_shard(i, 0) ? frame_span.size() / 4
                                                        : 0;
            const size_t bytes_written = shard.write_frame(frame_span, i);

            // we have 4 shards, so we should only have written 1/4 of the frame
            EXPECT(bytes_written == expected_bytes_written,
                   "Expected to write ",
                   expected_bytes_written,
                   " bytes, wrote ",
                   bytes_written,
                   " (frame ",
                   i,
                   ")");

            // should not have flushed
            CHECK(!fs::exists(shard_file_path));
        }

        // write the final frame in the shard
        {
            const size_t bytes_written =
              shard.write_frame(frame_span, frames_per_chunk - 1);
            const size_t expected_bytes_written =
              array_dimensions->frame_is_in_shard(frames_per_chunk - 1, 0)
                ? frame_span.size() / 4
                : 0;

            // we have 4 shards, so we should only have written 1/4 of the frame
            EXPECT(bytes_written == expected_bytes_written,
                   "Expected to write ",
                   expected_bytes_written,
                   " bytes, wrote ",
                   bytes_written);
        }
        EXPECT(shard.close(), "Shard did not close successfully");

        // should have flushed
        EXPECT(fs::is_regular_file(shard_file_path),
               "Shard should have flushed, did not");
        const auto actual_file_size = fs::file_size(shard_file_path);
        EXPECT(actual_file_size == expected_file_size,
               "Expected a file size of ",
               expected_file_size,
               ", got ",
               actual_file_size);

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR(e.what());
    }

    if (fs::exists(shard_file_path)) {
        fs::remove_all(shard_file_path);
    }

    return retval;
}