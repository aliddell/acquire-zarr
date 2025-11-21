#pragma once

#include "file.handle.hh"
#include "shard.hh"

namespace zarr {
class FSShard final : public Shard
{
  public:
    FSShard(ShardConfig&& config,
            std::shared_ptr<ThreadPool> thread_pool,
            std::shared_ptr<FileHandlePool> file_handle_pool);

  protected:
    std::shared_ptr<FileHandlePool> file_handle_pool_;
    std::shared_ptr<void> file_handle_;
    std::mutex handle_mutex_;

    bool write_to_offset_(const std::vector<uint8_t>& chunk,
                          size_t offset) override;

    void clean_up_resource_() override;
};
} // namespace zarr