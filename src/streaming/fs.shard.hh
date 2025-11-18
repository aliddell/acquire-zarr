#pragma once

#include "shard.hh"

namespace zarr {
class FSShard : public Shard
{
  public:
    FSShard(ShardConfig&& config, std::shared_ptr<ThreadPool> thread_pool);

  protected:
    bool flush_chunks_() override;
    bool compress_and_flush_chunks_() override;
};
} // namespace zarr