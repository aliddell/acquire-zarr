#include "acquire.zarr.h"
#include "test.macros.hh"

#include <nlohmann/json.hpp>

#include <fstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace {
const std::string test_path =
  (fs::temp_directory_path() / (TEST ".zarr")).string();

constexpr unsigned int array_width = 4096, array_height = 2304,
                       array_planes = 2560;

constexpr unsigned int chunk_width = 64, chunk_height = 64, chunk_planes = 32;

constexpr unsigned int shard_width = 1, shard_height = 1, shard_planes = 1;

constexpr size_t nbytes_px = sizeof(uint16_t);
constexpr uint32_t frames_to_acquire = array_planes;
constexpr size_t bytes_of_frame = array_width * array_height * nbytes_px;

ZarrStream*
setup()
{
    ZarrArraySettings array = {
        .data_type = ZarrDataType_uint16,
    };
    ZarrStreamSettings settings = {
        .store_path = test_path.c_str(),
        .s3_settings = nullptr,
        .max_threads = 0, // use all available threads
        .arrays = &array,
        .array_count = 1,
    };

    ZarrCompressionSettings compression_settings = {
        .compressor = ZarrCompressor_Blosc1,
        .codec = ZarrCompressionCodec_BloscLZ4,
        .level = 2,
        .shuffle = 2,
    };
    settings.arrays->compression_settings = &compression_settings;

    CHECK_OK(ZarrArraySettings_create_dimension_array(settings.arrays, 3));

    ZarrDimensionProperties* dim = settings.arrays->dimensions;
    *dim = DIM("z",
               ZarrDimensionType_Space,
               array_planes,
               chunk_planes,
               shard_planes,
               "millimeter",
               1.4);

    *(++dim) = DIM("y",
                   ZarrDimensionType_Space,
                   array_height,
                   chunk_height,
                   shard_height,
                   "micrometer",
                   0.9);

    *(++dim) = DIM("x",
                   ZarrDimensionType_Space,
                   array_width,
                   chunk_width,
                   shard_width,
                   "micrometer",
                   0.9);

    auto* stream = ZarrStream_create(&settings);
    ZarrArraySettings_destroy_dimension_array(settings.arrays);

    return stream;
}
} // namespace

int
main()
{
    Zarr_set_log_level(ZarrLogLevel_Info);

    auto* stream = setup();
    const std::vector<uint16_t> frame(array_width * array_height, 1);

    int retval = 1;

    try {
        size_t bytes_out;
        for (auto i = 0; i < frames_to_acquire; ++i) {
            if (i > 0 && i % 10 == 0) {
                LOG_INFO("Appending frame ", i, "/", frames_to_acquire);
            }

            const ZarrStatusCode status = ZarrStream_append(
              stream, frame.data(), bytes_of_frame, &bytes_out, nullptr);
            EXPECT(status == ZarrStatusCode_Success,
                   "Failed to append frame ",
                   i,
                   ": ",
                   Zarr_get_status_message(status));
            EXPECT_EQ(size_t, bytes_out, bytes_of_frame);
        }

        ZarrStream_destroy(stream);

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Caught exception: ", e.what());
    }

    // cleanup
    if (fs::exists(test_path)) {
        fs::remove_all(test_path);
    }

    return retval;
}
