#pragma once

#include "file.handle.hh"
#include "shard.hh"

namespace zarr {
class FSShard : public Shard
{
  public:
    FSShard(ShardConfig&& config,
            std::shared_ptr<ThreadPool> thread_pool,
            std::shared_ptr<FileHandlePool> file_handle_pool);

  protected:
    std::shared_ptr<FileHandlePool> file_handle_pool_;

    bool compress_and_flush_chunks_() override;
};
} // namespace zarr