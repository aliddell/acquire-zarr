#include "unit.test.macros.hh"
#include "zarr.stream.hh"

#include <nlohmann/json.hpp>

#include <istream>
#include <filesystem>

namespace fs = std::filesystem;

namespace {
const auto base_dir = (fs::temp_directory_path() / TEST ".zarr").string();

constexpr ZarrDataType data_type = ZarrDataType_float32;

constexpr uint64_t array_width = 1920, array_height = 1080, array_planes = 32;
constexpr uint64_t chunk_width = 128, chunk_height = 128, chunk_planes = 32;
constexpr uint64_t shard_width = (1920 + 129) / 128,
                   shard_height = (1080 + 129) / 128, shard_planes = 1;

constexpr size_t frame_size_px = array_width * array_height;
constexpr size_t frame_size_bytes = frame_size_px * sizeof(float);

const std::vector frame(frame_size_px, 1.0f);

void
configure_stream_dimensions(ZarrArraySettings* settings)
{
    CHECK(ZarrStatusCode_Success ==
          ZarrArraySettings_create_dimension_array(settings, 3));

    settings->data_type = data_type;

    ZarrDimensionProperties* dim = settings->dimensions;
    *(dim++) = ZarrDimensionProperties{
        .name = "z",
        .array_size_px = array_planes,
        .chunk_size_px = chunk_planes,
        .shard_size_chunks = shard_planes,
    };

    *(dim++) = ZarrDimensionProperties{
        .name = "y",
        .array_size_px = array_height,
        .chunk_size_px = chunk_height,
        .shard_size_chunks = shard_height,
    };

    *dim = ZarrDimensionProperties{
        .name = "z",
        .array_size_px = array_width,
        .chunk_size_px = chunk_width,
        .shard_size_chunks = shard_width,
    };
}

ZarrStream*
make_stream()
{
    ZarrStreamSettings settings{
        .store_path = base_dir.c_str(),
        .overwrite = true,
    };

    CHECK(ZarrStatusCode_Success ==
          ZarrStreamSettings_create_arrays(&settings, 1));

    configure_stream_dimensions(settings.arrays);

    ZarrStream* stream = ZarrStream_create(&settings);
    EXPECT(stream != nullptr, "Failed to create stream.");

    ZarrStreamSettings_destroy_arrays(&settings);

    return stream;
}

void
test_write_wrong_key_fails(ZarrStream* stream)
{
    size_t bytes_out;
    CHECK(ZarrStatusCode_KeyNotFound ==
          stream->append_frame(
            "foo", frame.data(), 0, nullptr, frame_size_bytes, bytes_out));
    CHECK(bytes_out == 0);
}

void
test_write_zero_bytes(ZarrStream* stream)
{
    size_t bytes_out;
    CHECK(ZarrStatusCode_Success ==
          stream->append_frame(nullptr, nullptr, 0, nullptr, 0, bytes_out));
    CHECK(bytes_out == 0);
}

void
test_write_too_few_bytes_fails(ZarrStream* stream)
{
    size_t bytes_out;
    CHECK(
      ZarrStatusCode_InvalidArgument ==
      stream->append_frame(
        nullptr, frame.data(), 0, nullptr, frame_size_bytes - 1, bytes_out));
    CHECK(bytes_out == 0);
}

void
test_write_too_many_bytes_fails(ZarrStream* stream)
{
    size_t bytes_out;
    CHECK(
      ZarrStatusCode_InvalidArgument ==
      stream->append_frame(
        nullptr, frame.data(), 0, nullptr, frame_size_bytes + 1, bytes_out));
    CHECK(bytes_out == 0);
}

void
test_nonempty_frame_buffer_fails(ZarrStream* stream)
{
    size_t bytes_out;

    // append a single byte
    CHECK(ZarrStatusCode_Success ==
          stream->append(nullptr, frame.data(), 1, bytes_out));
    CHECK(bytes_out == 1);

    // try to append the frame
    CHECK(ZarrStatusCode_WillNotOverwrite ==
          stream->append_frame(
            nullptr, frame.data(), 0, nullptr, frame_size_bytes, bytes_out));
    CHECK(bytes_out == 0);

    // append a single byte
    CHECK(
      ZarrStatusCode_Success ==
      stream->append(nullptr, frame.data(), frame_size_bytes - 1, bytes_out));
    CHECK(bytes_out == frame_size_bytes - 1);
}

void
test_write_frame(ZarrStream* stream)
{
    size_t bytes_out;
    CHECK(ZarrStatusCode_Success ==
          stream->append_frame(
            nullptr, frame.data(), 1, nullptr, frame_size_bytes, bytes_out));
    CHECK(bytes_out == frame_size_bytes);
}

void
test_out_of_bounds(ZarrStream* stream)
{
    // append the rest of the 32 frames
    size_t bytes_out;
    for (auto i = 2; i < 32; ++i) {
        CHECK(
          ZarrStatusCode_Success ==
          stream->append_frame(
            nullptr, frame.data(), i, nullptr, frame_size_bytes, bytes_out));
        CHECK(bytes_out == frame_size_bytes);
    }

    // try to append out of bounds
    CHECK(ZarrStatusCode_WriteOutOfBounds ==
          stream->append_frame(
            nullptr, frame.data(), 32, nullptr, frame_size_bytes, bytes_out));
    CHECK(bytes_out == 0);
}
} // namespace

int
main()
{
    int retval = 1;
    ZarrStream* stream = nullptr;

    try {
        stream = make_stream();
        test_write_wrong_key_fails(stream);
        test_write_too_few_bytes_fails(stream);
        test_write_too_many_bytes_fails(stream);
        test_nonempty_frame_buffer_fails(stream); // 1 frame written
        test_write_zero_bytes(stream);
        test_write_frame(stream);   // 1 frame written (2 total)
        test_out_of_bounds(stream); // 30 frames written (32 total)

        retval = 0;
    } catch (const std::exception& exc) {
        LOG_ERROR("Error: ", exc.what());
    }

    if (stream) {
        ZarrStream_destroy(stream);
    }

    if (fs::exists(base_dir)) {
        std::error_code ec;
        fs::remove_all(base_dir, ec);
        if (ec) {
            LOG_ERROR("Failed to remove ", base_dir, ": ", ec.message());
        }
    }

    return retval;
}
