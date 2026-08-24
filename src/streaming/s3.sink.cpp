#include "macros.hh"
#include "s3.sink.hh"

#include <utility>

zarr::S3Sink::S3Sink(std::string_view bucket_name,
                     std::string_view object_key,
                     std::shared_ptr<S3Client> client)
  : bucket_name_{ bucket_name }
  , object_key_{ object_key }
  , client_{ std::move(client) }
{
    EXPECT(!bucket_name_.empty(), "Bucket name must not be empty");
    EXPECT(!object_key_.empty(), "Object key must not be empty");
    EXPECT(client_, "Null pointer: client");
}

void
zarr::S3Sink::coalesce_()
{
    for (auto it = pending_.find(nbytes_appended_ + staged_.size());
         it != pending_.end();
         it = pending_.find(nbytes_appended_ + staged_.size())) {
        staged_.insert(staged_.end(), it->second.begin(), it->second.end());
        pending_.erase(it);
    }
}

bool
zarr::S3Sink::drain_(std::unique_lock<std::mutex>& lock)
{
    if (draining_) {
        // another writer is appending and will pick up what we staged
        return true;
    }

    draining_ = true;
    bool retval = true;

    while (retval && !staged_.empty() &&
           (upload_ || staged_.size() >= min_part_size_)) {
        if (!upload_) {
            lock.unlock();
            auto upload = client_->create_upload(bucket_name_, object_key_);
            lock.lock();

            if (!upload) {
                LOG_ERROR("Failed to start upload of object ", object_key_);
                retval = false;
                break;
            }
            upload_ = std::move(upload);
        }

        auto batch = std::move(staged_);
        staged_.clear();
        nbytes_appended_ += batch.size();

        lock.unlock();
        retval = upload_->append(batch);
        lock.lock();
    }

    draining_ = false;
    if (!retval) {
        failed_ = true;
    }
    cv_.notify_all();

    return retval;
}

bool
zarr::S3Sink::write(size_t offset, ConstByteSpan data)
{
    if (data.data() == nullptr || data.empty()) {
        return true;
    }

    std::unique_lock lock(mutex_);

    if (failed_) {
        return false;
    }

    if (offset < nbytes_appended_) {
        LOG_ERROR("Cannot write data at offset ",
                  offset,
                  " of object ",
                  object_key_,
                  ", already uploaded through ",
                  nbytes_appended_);
        return false;
    }

    // Shard::write_chunk replays the same offset when a write is retried, so a
    // fragment we are still holding is a no-op rather than an error
    if (const auto it = pending_.find(offset); it != pending_.end()) {
        if (it->second.size() != data.size()) {
            LOG_ERROR("Retried write at offset ",
                      offset,
                      " of object ",
                      object_key_,
                      " changed size from ",
                      it->second.size(),
                      " to ",
                      data.size());
            return false;
        }
    } else {
        pending_.emplace(offset, ByteVector(data.begin(), data.end()));
    }

    coalesce_();

    return drain_(lock);
}

bool
zarr::S3Sink::flush_()
{
    std::unique_lock lock(mutex_);

    // finalization should follow the last write, but don't touch the upload
    // underneath a writer that is still appending
    cv_.wait(lock, [this] { return !draining_; });

    if (failed_) {
        return false;
    }

    if (!pending_.empty()) {
        LOG_ERROR("Object ",
                  object_key_,
                  " is missing the bytes at offset ",
                  nbytes_appended_ + staged_.size(),
                  "; ",
                  pending_.size(),
                  " fragment(s) cannot be uploaded");
        return false;
    }

    if (!upload_) {
        if (staged_.empty()) {
            return true;
        }

        // never spilled, so send it in one request rather than opening a
        // multipart upload for it
        auto batch = std::move(staged_);
        staged_.clear();
        nbytes_appended_ += batch.size();

        lock.unlock();
        return client_->put_object(bucket_name_, object_key_, batch);
    }

    while (!staged_.empty()) {
        auto batch = std::move(staged_);
        staged_.clear();
        nbytes_appended_ += batch.size();

        lock.unlock();
        const bool appended = upload_->append(batch);
        lock.lock();

        if (!appended) {
            failed_ = true;
            return false;
        }
    }

    auto* upload = upload_.get();
    lock.unlock();

    return upload->finish();
}
