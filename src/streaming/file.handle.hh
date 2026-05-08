#pragma once

#include <condition_variable>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace zarr {
/**
 * @brief A handle to a file, wrapping the platform-specific file handle.
 * @details This class is not copyable or movable. It is intended to be used
 * with FileHandlePool to manage a pool of file handles.
 */
class FileHandle
{
  public:
    FileHandle(const std::string& filename, bool aligned);
    ~FileHandle(); // calls destroy_handle

    void* get() const;
    bool is_aligned() const { return aligned_; }

    // not copyable or movable
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    FileHandle(FileHandle&&) = delete;
    FileHandle& operator=(FileHandle&&) = delete;

  private:
    void* handle_;
    bool aligned_;
};

class FileHandlePool; // forward decl

struct BorrowedHandle
{
    BorrowedHandle() = default;
    BorrowedHandle(FileHandle* handle,
                   std::string filename,
                   FileHandlePool* pool)
      : handle_(handle)
      , filename_(std::move(filename))
      , pool_(pool)
    {
    }
    ~BorrowedHandle(); // calls pool_->return_handle(filename_)

    // movable, not copyable
    BorrowedHandle(const BorrowedHandle&) = delete;
    BorrowedHandle& operator=(const BorrowedHandle&) = delete;
    BorrowedHandle(BorrowedHandle&&) = default;
    BorrowedHandle& operator=(BorrowedHandle&&) = default;

    FileHandle* handle_ = nullptr;
    std::string filename_;
    FileHandlePool* pool_ = nullptr;
};

/**
 * @brief A pool of file handles to limit the number of concurrently open files.
 */
class FileHandlePool
{
  public:
    FileHandlePool();
    ~FileHandlePool() = default;

    BorrowedHandle get_handle(const std::string& filename, bool aligned);
    void return_handle(const std::string& filename);

    uint64_t io_alignment() const { return io_alignment_; }

  private:
    struct CacheEntry
    {
        std::shared_ptr<FileHandle> handle;
        std::list<std::string>::iterator lru_it;
        uint32_t refcount = 0;
    };

    const uint64_t max_active_handles_;
    const uint64_t io_alignment_;
    std::list<std::string> lru_order_; // front = most recent
    std::unordered_map<std::string, CacheEntry> cache_;
    bool cache_space_available_;
    std::mutex mutex_;
    std::condition_variable cv_;

    // Evicts the least recently used handle with refcount == 0, if any.
    // Must be called with mutex_ held.
    void evict_lru_();
};
} // namespace zarr