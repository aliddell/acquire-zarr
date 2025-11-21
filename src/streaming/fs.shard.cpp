#include "fs.shard.hh"
#include "macros.hh"

#include <filesystem>
#include <unordered_set>

bool
seek_and_write(void* handle, size_t offset, std::span<const uint8_t> data);

namespace fs = std::filesystem;

namespace {
std::vector<std::string>
get_parent_paths(const std::vector<std::string>& file_paths)
{
    std::unordered_set<std::string> unique_paths;
    for (const auto& file_path : file_paths) {
        unique_paths.emplace(fs::path(file_path).parent_path().string());
    }

    return { unique_paths.begin(), unique_paths.end() };
}

bool
make_dirs(const std::vector<std::string>& dir_paths,
          std::shared_ptr<zarr::ThreadPool> thread_pool)
{
    if (dir_paths.empty()) {
        return true;
    }
    EXPECT(thread_pool, "Thread pool not provided.");

    std::atomic<char> all_successful = 1;
    const std::unordered_set unique_paths(dir_paths.begin(), dir_paths.end());

    std::vector<std::future<void>> futures;

    for (const auto& path : unique_paths) {
        auto promise = std::make_shared<std::promise<void>>();
        futures.emplace_back(promise->get_future());

        auto job = [path, promise, &all_successful](std::string& err) {
            bool success = true;
            try {
                if (fs::is_directory(path) || path.empty()) {
                    promise->set_value();
                    return success;
                }

                std::error_code ec;
                if (!fs::create_directories(path, ec) &&
                    !fs::is_directory(path)) {
                    err = "Failed to create directory '" + path +
                          "': " + ec.message();
                    success = false;
                }
            } catch (const std::exception& exc) {
                err =
                  "Failed to create directory '" + path + "': " + exc.what();
                success = false;
            }

            promise->set_value();
            all_successful.fetch_and(success);
            return success;
        };

        if (thread_pool->n_threads() == 1 || !thread_pool->push_job(job)) {
            if (std::string err; !job(err)) {
                LOG_ERROR(err);
            }
        }
    }

    // wait for all jobs to finish
    for (auto& future : futures) {
        future.wait();
    }

    return all_successful;
}
} // namespace

zarr::FSShard::FSShard(ShardConfig&& config,
                       std::shared_ptr<ThreadPool> thread_pool,
                       std::shared_ptr<FileHandlePool> file_handle_pool)
  : Shard(std::move(config), thread_pool)
  , file_handle_pool_(file_handle_pool)
{
}

bool
zarr::FSShard::write_to_offset_(const std::vector<uint8_t>& chunk,
                                size_t offset)
{
    {
        std::unique_lock lock(handle_mutex_);
        if (file_handle_ == nullptr) {
            file_handle_ = file_handle_pool_->get_handle(config_.path);
        }
    }

    return seek_and_write(file_handle_.get(), offset, chunk);
}

void
zarr::FSShard::clean_up_resource_()
{
    file_handle_pool_->close_handle(config_.path);
    file_handle_.reset();
}
