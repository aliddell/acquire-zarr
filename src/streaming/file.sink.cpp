#include "file.sink.hh"
#include "macros.hh"

#include <string_view>

bool
seek_and_write(void* handle, size_t offset, ConstByteSpan data);

bool
seek_and_write_aligned(void* handle, size_t offset, ConstByteSpan data);

bool
flush_file(void* handle);

zarr::FileSink::FileSink(std::string_view filename,
                         std::shared_ptr<FileHandlePool> file_handle_pool,
                         bool aligned)
  : file_handle_pool_(file_handle_pool)
  , filename_(filename)
  , aligned_(aligned)
{
    EXPECT(file_handle_pool_ != nullptr, "File handle pool not provided.");
}

bool
zarr::FileSink::write(size_t offset, ConstByteSpan data)
{
    if (data.data() == nullptr || data.size() == 0) {
        return true;
    }

    const auto borrowed = file_handle_pool_->get_handle(filename_, aligned_);
    if (borrowed.handle_ == nullptr) {
        LOG_ERROR("Failed to get file handle for ", filename_);
        return false;
    }

    bool retval = false;
    try {
        if (borrowed.handle_->is_aligned()) {
            retval = seek_and_write_aligned(borrowed.handle_->get(), offset, data);
        } else {
            retval = seek_and_write(borrowed.handle_->get(), offset, data);
        }
    } catch (const std::exception& exc) {
        LOG_ERROR("Failed to write to file ", filename_, ": ", exc.what());
    }

    return retval;
    // borrowed goes out of scope here, return_handle called automatically
}

size_t
zarr::FileSink::io_alignment() const
{
    return aligned_ ? file_handle_pool_->io_alignment() : 1;
}

bool
zarr::FileSink::flush_()
{
    const auto borrowed = file_handle_pool_->get_handle(filename_, aligned_);
    if (borrowed.handle_ == nullptr) {
        LOG_ERROR("Failed to get file handle for ", filename_);
        return false;
    }

    bool retval = false;
    try {
        retval = flush_file(borrowed.handle_->get());
    } catch (const std::exception& exc) {
        LOG_ERROR("Failed to flush file ", filename_, ": ", exc.what());
    }

    return retval;
}