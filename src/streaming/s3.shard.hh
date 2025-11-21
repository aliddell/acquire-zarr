#pragma once

#include "s3.connection.hh"
#include "shard.hh"

namespace zarr {
class S3Shard final : public Shard
{
  public:
    S3Shard(ShardConfig&& config,
            std::shared_ptr<ThreadPool> thread_pool,
            std::shared_ptr<S3ConnectionPool> s3_connection_pool);

  protected:
    std::shared_ptr<S3ConnectionPool> s3_connection_pool_;

    bool write_to_offset_(const std::vector<uint8_t>& chunk,
                          size_t offset) override;

    void clean_up_resource_() override;
};
} // namespace zarr