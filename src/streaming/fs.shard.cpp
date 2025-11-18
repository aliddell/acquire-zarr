#include "fs.shard.hh"

zarr::FSShard::FSShard(ShardConfig&& config,
                       std::shared_ptr<ThreadPool> thread_pool,
                       std::shared_ptr<FileHandlePool> file_handle_pool)
  : Shard(std::move(config), thread_pool)
  , file_handle_pool_(file_handle_pool)
{
}

bool
zarr::FSShard::compress_and_flush_chunks_()
{
    return false;
}
