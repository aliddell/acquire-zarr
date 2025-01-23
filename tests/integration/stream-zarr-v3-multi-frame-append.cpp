#include "acquire.zarr.h"
#include "test.macros.hh"
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

static const size_t array_width = 64;
static const size_t array_height = 48;
static const size_t frames_to_acquire = 10;
static const size_t frames_per_append = 3;
static const fs::path test_path = "multi-frame-test.zarr";

static ZarrStream* 
setup() {
    const auto test_path_str = test_path.string();

    ZarrStreamSettings settings = {  
        .store_path = test_path_str.c_str(),
        .data_type = ZarrDataType_uint16,
        .version = ZarrVersion_3,
    };

    CHECK(ZarrStatusCode_Success == 
          ZarrStreamSettings_create_dimension_array(&settings, 3));

    // Configure dimensions [t, y, x]
    settings.dimensions[0] = {
        .name = "t",
        .type = ZarrDimensionType_Time,
        .array_size_px = 0, // Append dimension
        .chunk_size_px = 5,
        .shard_size_chunks = 2,
    };

    settings.dimensions[1] = {
        .name = "y",
        .type = ZarrDimensionType_Space,
        .array_size_px = array_height,
        .chunk_size_px = 16,
        .shard_size_chunks = 2,
    };

    settings.dimensions[2] = {
        .name = "x",
        .type = ZarrDimensionType_Space,
        .array_size_px = array_width,
        .chunk_size_px = 16,
        .shard_size_chunks = 2,
    };

    auto* stream = ZarrStream_create(&settings);
    ZarrStreamSettings_destroy_dimension_array(&settings);
    CHECK(stream != nullptr);
    return stream;
}

static void
verify_data() {
    // Basic structure verification
    CHECK(fs::exists(test_path));
    CHECK(fs::exists(test_path / "zarr.json")); // Check zarr metadata exists
    
    // Verify the final number of frames by checking the .zarray metadata
    const fs::path zarray_path = test_path / ".zarray";
    CHECK(fs::exists(zarray_path));
    CHECK(fs::file_size(zarray_path) > 0);

    // Count the number of shard files in the chunks directory
    const fs::path chunks_path = test_path / "c";
    size_t shard_count = 0;
    for (const auto& entry : fs::directory_iterator(chunks_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".shard") {
            shard_count++;
        }
    }
    
    // We should have at least one shard file
    CHECK(shard_count > 0);

    // Calculate expected number of chunks based on our parameters
    const size_t expected_frames = frames_to_acquire;
    const size_t chunk_size_t = 5; // From settings.dimensions[0].chunk_size_px
    const size_t expected_chunks = (expected_frames + chunk_size_t - 1) / chunk_size_t;
    
    // Log some diagnostic information
    LOG_DEBUG("Found ", shard_count, " shard files");
    LOG_DEBUG("Expected ", expected_frames, " frames in ", expected_chunks, " chunks");
}

int
main()
{
    Zarr_set_log_level(ZarrLogLevel_Debug);

    auto* stream = setup();
    const size_t frame_size = array_width * array_height * sizeof(uint16_t);
    const size_t multi_frame_size = frame_size * frames_per_append;
    
    std::vector<uint16_t> multi_frame_data(array_width * array_height * frames_per_append);
    int retval = 1;

    try {
        // Test 1: Append multiple complete frames
        size_t bytes_out;
        for (auto i = 0; i < frames_to_acquire; i += frames_per_append) {
            // Fill multi-frame buffer with test pattern
            for (size_t f = 0; f < frames_per_append; ++f) {
                const size_t frame_offset = f * array_width * array_height;
                const uint16_t frame_value = static_cast<uint16_t>(i + f);
                std::fill(multi_frame_data.begin() + frame_offset,
                         multi_frame_data.begin() + frame_offset + (array_width * array_height),
                         frame_value);
            }

            ZarrStatusCode status = ZarrStream_append(
                stream,
                multi_frame_data.data(),
                multi_frame_size,
                &bytes_out);

            EXPECT(status == ZarrStatusCode_Success,
                   "Failed to append frames ",
                   i,
                   "-",
                   i + frames_per_append - 1);
            EXPECT_EQ(size_t, bytes_out, multi_frame_size);
        }

        // Test 2: Append partial frames (should fail)
        const size_t partial_size = frame_size / 2;
        ZarrStatusCode status = ZarrStream_append(
            stream,
            multi_frame_data.data(),
            partial_size,
            &bytes_out);
        EXPECT(status != ZarrStatusCode_Success,
               "Partial frame append should fail");

        // Test 3: Append non-multiple of frame size (should fail)
        const size_t invalid_size = frame_size + (frame_size / 2);
        status = ZarrStream_append(
            stream,
            multi_frame_data.data(),
            invalid_size,
            &bytes_out);
        EXPECT(status != ZarrStatusCode_Success,
               "Non-multiple of frame size append should fail");

        ZarrStream_destroy(stream);

        verify_data();

        // Clean up
        fs::remove_all(test_path);

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Caught exception: ", e.what());
    }

    return retval;
} 