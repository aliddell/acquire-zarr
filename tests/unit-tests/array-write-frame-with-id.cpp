#include "array.hh"
#include "unit.test.macros.hh"
#include "zarr.common.hh"

#include <filesystem>

namespace fs = std::filesystem;

namespace {
const fs::path base_dir = fs::temp_directory_path() / TEST;

constexpr unsigned int array_width = 64, array_height = 48,
                       array_timepoints = 10;
constexpr unsigned int chunk_width = 16, chunk_height = 16,
                       chunk_timepoints = 5;
constexpr unsigned int n_frames = array_timepoints;

std::unique_ptr<zarr::Array>
make_array(std::shared_ptr<zarr::ThreadPool> thread_pool)
{
    std::vector<ZarrDimension> dims;
    dims.emplace_back(
      "t", ZarrDimensionType_Time, array_timepoints, chunk_timepoints, 1);
    dims.emplace_back(
      "y", ZarrDimensionType_Space, array_height, chunk_height, 1);
    dims.emplace_back(
      "x", ZarrDimensionType_Space, array_width, chunk_width, 1);

    auto config = std::make_shared<zarr::ArrayConfig>(
      base_dir.string(),
      "",
      std::nullopt,
      std::nullopt,
      std::make_shared<ArrayDimensions>(std::move(dims), ZarrDataType_uint16),
      ZarrDataType_uint16,
      std::nullopt,
      0,
      false,
      false);

    return std::make_unique<zarr::Array>(
      config, thread_pool, std::make_shared<zarr::FileHandlePool>(), nullptr);
}
} // namespace

void
test_happy_path_no_frame_ids()
{
    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      std::thread::hardware_concurrency(),
      [](const std::string& err) { LOG_ERROR("Error: ", err); });

    auto writer = make_array(thread_pool);
    const size_t frame_size =
      array_width * array_height * zarr::bytes_of_type(ZarrDataType_uint16);
    zarr::LockedBuffer data(std::move(ByteVector(frame_size, 0)));

    for (auto i = 0; i < n_frames; ++i) {
        size_t bytes_out;
        CHECK(
          writer->write_frame(data, bytes_out, std::nullopt, std::nullopt) ==
          zarr::WriteResult::Ok);
        CHECK(bytes_out == data.size());
    }
}

void
test_happy_path_with_frame_ids()
{
    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      std::thread::hardware_concurrency(),
      [](const std::string& err) { LOG_ERROR("Error: ", err); });

    auto writer = make_array(thread_pool);
    const size_t frame_size =
      array_width * array_height * zarr::bytes_of_type(ZarrDataType_uint16);
    zarr::LockedBuffer data(std::move(ByteVector(frame_size, 0)));

    // Start at an arbitrary offset to verify any start value is accepted
    const uint64_t start_id = 42;
    for (auto i = 0; i < n_frames; ++i) {
        size_t bytes_out;
        CHECK(
          writer->write_frame(data, bytes_out, start_id + i, std::nullopt) ==
          zarr::WriteResult::Ok);
        CHECK(bytes_out == data.size());
    }
}

void
test_out_of_sequence_frame_id()
{
    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      std::thread::hardware_concurrency(),
      [](const std::string& err) { LOG_ERROR("Error: ", err); });

    auto writer = make_array(thread_pool);
    const size_t frame_size =
      array_width * array_height * zarr::bytes_of_type(ZarrDataType_uint16);
    zarr::LockedBuffer data(std::move(ByteVector(frame_size, 0)));

    size_t bytes_out;
    CHECK(writer->write_frame(data, bytes_out, 0, std::nullopt) ==
          zarr::WriteResult::Ok);
    CHECK(writer->write_frame(data, bytes_out, 1, std::nullopt) ==
          zarr::WriteResult::Ok);

    // skip frame ID 2
    CHECK(writer->write_frame(data, bytes_out, 3, std::nullopt) ==
          zarr::WriteResult::SkippedFrame);

    // array should still be usable -- correct next ID succeeds
    CHECK(writer->write_frame(data, bytes_out, 2, std::nullopt) ==
          zarr::WriteResult::Ok);
}

void
test_missing_frame_id()
{
    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      std::thread::hardware_concurrency(),
      [](const std::string& err) { LOG_ERROR("Error: ", err); });

    auto writer = make_array(thread_pool);
    const size_t frame_size =
      array_width * array_height * zarr::bytes_of_type(ZarrDataType_uint16);
    zarr::LockedBuffer data(std::move(ByteVector(frame_size, 0)));

    size_t bytes_out;
    CHECK(writer->write_frame(data, bytes_out, 0, std::nullopt) ==
          zarr::WriteResult::Ok);

    // drop the frame ID
    CHECK(writer->write_frame(data, bytes_out, std::nullopt, std::nullopt) ==
          zarr::WriteResult::MissingFrameId);

    // array should still be usable
    CHECK(writer->write_frame(data, bytes_out, 1, std::nullopt) ==
          zarr::WriteResult::Ok);
}

void
test_unexpected_frame_id()
{
    auto thread_pool = std::make_shared<zarr::ThreadPool>(
      std::thread::hardware_concurrency(),
      [](const std::string& err) { LOG_ERROR("Error: ", err); });

    auto writer = make_array(thread_pool);
    const size_t frame_size =
      array_width * array_height * zarr::bytes_of_type(ZarrDataType_uint16);
    zarr::LockedBuffer data(std::move(ByteVector(frame_size, 0)));

    size_t bytes_out;
    CHECK(writer->write_frame(data, bytes_out, std::nullopt, std::nullopt) ==
          zarr::WriteResult::Ok);

    // introduce a frame ID after an un-ID'd frame
    CHECK(writer->write_frame(data, bytes_out, 0, std::nullopt) ==
          zarr::WriteResult::UnexpectedFrameId);

    // array should still be usable
    CHECK(writer->write_frame(data, bytes_out, std::nullopt, std::nullopt) ==
          zarr::WriteResult::Ok);
}

int
main()
{
    Logger::set_log_level(LogLevel_Debug);

    int retval = 1;

    try {
        test_happy_path_no_frame_ids();
        if (fs::exists(base_dir)) {
            fs::remove_all(base_dir);
        }

        test_happy_path_with_frame_ids();
        if (fs::exists(base_dir)) {
            fs::remove_all(base_dir);
        }

        test_out_of_sequence_frame_id();
        if (fs::exists(base_dir)) {
            fs::remove_all(base_dir);
        }

        test_missing_frame_id();
        if (fs::exists(base_dir)) {
            fs::remove_all(base_dir);
        }

        test_unexpected_frame_id();
        if (fs::exists(base_dir)) {
            fs::remove_all(base_dir);
        }

        retval = 0;
    } catch (const std::exception& exc) {
        LOG_ERROR("Exception: ", exc.what());
    }

    if (fs::exists(base_dir)) {
        fs::remove_all(base_dir);
    }

    return retval;
}