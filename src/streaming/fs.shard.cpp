#include "fs.shard.hh"

zarr::FSShard::FSShard(ShardConfig&& config,
                       std::shared_ptr<ThreadPool> thread_pool)
  : Shard(std::move(config), thread_pool)
{
}

bool
zarr::FSShard::flush_chunks_()
{
    return false;
}
