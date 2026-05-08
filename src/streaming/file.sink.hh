#pragma once

#include "file.handle.hh"
#include "sink.hh"

#include <fstream>
#include <mutex>
#include <string_view>

namespace zarr {
class FileSink : public Sink
{
  public:
    FileSink(std::string_view filename,
             std::shared_ptr<FileHandlePool> file_handle_pool,
             bool aligned);
    ~FileSink() override = default;

    bool write(size_t offset, ConstByteSpan data) override;
    size_t io_alignment() const override;

  protected:
    bool flush_() override;

  private:
    std::shared_ptr<FileHandlePool> file_handle_pool_;
    std::string filename_;
    bool aligned_;
};
} // namespace zarr
