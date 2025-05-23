#pragma once

#include "array.hh"
#include "definitions.hh"
#include "downsampler.hh"
#include "frame.queue.hh"
#include "group.hh"
#include "s3.connection.hh"
#include "sink.hh"
#include "thread.pool.hh"
#include "array.dimensions.hh"

#include <nlohmann/json.hpp>

#include <condition_variable>
#include <cstddef> // size_t
#include <memory>  // unique_ptr
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

struct ZarrStream_s
{
  public:
    ZarrStream_s(struct ZarrStreamSettings_s* settings);

    /**
     * @brief Append data to the stream.
     * @param data The data to append.
     * @param nbytes The number of bytes to append.
     * @return The number of bytes appended.
     */
    size_t append(const void* data, size_t nbytes);

    /**
     * @brief Write custom metadata to the stream.
     * @param custom_metadata JSON-formatted custom metadata to write.
     * @param overwrite If true, overwrite any existing custom metadata.
     * Otherwise, fail if custom metadata has already been written.
     * @return ZarrStatusCode_Success on success, or an error code on failure.
     */
    ZarrStatusCode write_custom_metadata(std::string_view custom_metadata,
                                         bool overwrite);

  private:
    std::string error_; // error message. If nonempty, an error occurred.

    ZarrVersion version_;
    std::string store_path_;
    std::string output_key_;
    std::optional<zarr::S3Settings> s3_settings_;

    std::unique_ptr<zarr::ZarrNode> output_node_;

    size_t frame_size_bytes_;
    std::vector<std::byte> frame_buffer_;
    size_t frame_buffer_offset_;

    std::atomic<bool> process_frames_{ true };
    std::mutex frame_queue_mutex_;
    std::condition_variable frame_queue_not_full_cv_;  // Space is available
    std::condition_variable frame_queue_not_empty_cv_; // Data is available
    std::condition_variable frame_queue_empty_cv_;     // Queue is empty
    std::condition_variable frame_queue_finished_cv_;  // Done processing
    std::unique_ptr<zarr::FrameQueue> frame_queue_;

    std::shared_ptr<zarr::ThreadPool> thread_pool_;
    std::shared_ptr<zarr::S3ConnectionPool> s3_connection_pool_;

    std::unique_ptr<zarr::Sink> custom_metadata_sink_;

    bool is_s3_acquisition_() const;

    /**
     * @brief Check that the settings are valid.
     * @note Sets the error_ member if settings are invalid.
     * @param settings Struct containing settings to validate.
     * @return true if settings are valid, false otherwise.
     */
    [[nodiscard]] bool validate_settings_(const struct ZarrStreamSettings_s* settings);

    /**
     * @brief Configure the stream for a group.
     * @param settings Struct containing settings to configure.
     */
    [[nodiscard]] bool configure_group_(const struct ZarrStreamSettings_s* settings);

    /**
     * @brief Configure the stream for an array.
     * @param settings Struct containing settings to configure.
     */
    [[nodiscard]] bool configure_array_(const struct ZarrStreamSettings_s* settings);

    /**
     * @brief Copy settings to the stream and create the output node.
     * @param settings Struct containing settings to copy.
     * @return True if the output node was created successfully, false otherwise.
     */
    [[nodiscard]] bool commit_settings_(const struct ZarrStreamSettings_s* settings);

    /**
     * @brief Spin up the thread pool.
     */
    void start_thread_pool_(uint32_t max_threads);

    /**
     * @brief Set an error message.
     * @param msg The error message to set.
     */
    void set_error_(const std::string& msg);

    /**
     * @brief Create the data store.
     * @param overwrite Delete everything in the store path if true.
     * @return Return True if the store was created successfully, otherwise
     * false.
     */
    [[nodiscard]] bool create_store_(bool overwrite);

    [[nodiscard]] bool write_intermediate_metadata_();

    /** @brief Initialize the frame queue. */
    [[nodiscard]] bool init_frame_queue_();

    /** @brief Process the frame queue. */
    void process_frame_queue_();

    /** @brief Wait for the frame queue to finish processing. */
    void finalize_frame_queue_();

    friend bool finalize_stream(struct ZarrStream_s* stream);
};

bool
finalize_stream(struct ZarrStream_s* stream);
