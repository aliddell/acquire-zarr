#include "s3.shard.hh"

zarr::S3Shard::S3Shard(ShardConfig&& config,
                       std::shared_ptr<ThreadPool> thread_pool,
                       std::shared_ptr<S3ConnectionPool> s3_connection_pool)
  : Shard(std::move(config), thread_pool)
  , s3_connection_pool_(s3_connection_pool)
{
}

bool
zarr::S3Shard::write_to_offset_(const std::vector<uint8_t>& chunk,
                                size_t offset)
{
    return false;
}

void
zarr::S3Shard::clean_up_resource_()
{
}
