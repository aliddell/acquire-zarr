#pragma once

#include "array.writer.hh"

namespace zarr {
class ZarrV2ArrayWriter final : public ArrayWriter
{
  public:
    ZarrV2ArrayWriter(const ArrayWriterConfig& config,
                      std::shared_ptr<ThreadPool> thread_pool);

    ZarrV2ArrayWriter(const ArrayWriterConfig& config,
                      std::shared_ptr<ThreadPool> thread_pool,
                      std::shared_ptr<S3ConnectionPool> s3_connection_pool);

  private:
    std::string data_root_() const override;
    std::string metadata_path_() const override;
    const DimensionPartsFun parts_along_dimension_() const override;
    bool compress_and_flush_data_() override;
    bool write_array_metadata_() override;
    bool should_rollover_() const override;
};
} // namespace zarr
