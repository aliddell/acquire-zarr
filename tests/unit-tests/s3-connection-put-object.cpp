#include "s3.connection.hh"
#include "unit.test.macros.hh"

#include <cstdlib>
#include <optional>

namespace {
bool
get_credentials(std::string& endpoint,
                std::string& bucket_name,
                std::string& access_key_id,
                std::string& secret_access_key,
                std::optional<std::string>& region)
{
    char* env = nullptr;
    if (!(env = std::getenv("ZARR_S3_ENDPOINT"))) {
        LOG_ERROR("ZARR_S3_ENDPOINT not set.");
        return false;
    }
    endpoint = env;

    if (!(env = std::getenv("ZARR_S3_BUCKET_NAME"))) {
        LOG_ERROR("ZARR_S3_BUCKET_NAME not set.");
        return false;
    }
    bucket_name = env;

    if (!(env = std::getenv("ZARR_S3_ACCESS_KEY_ID"))) {
        LOG_ERROR("ZARR_S3_ACCESS_KEY_ID not set.");
        return false;
    }
    access_key_id = env;

    if (!(env = std::getenv("ZARR_S3_SECRET_ACCESS_KEY"))) {
        LOG_ERROR("ZARR_S3_SECRET_ACCESS_KEY not set.");
        return false;
    }
    secret_access_key = env;

    env = std::getenv("ZARR_S3_REGION");
    if (env) {
        region = env;
    }

    return true;
}
} // namespace

int
main()
{
    std::string s3_endpoint, bucket_name, s3_access_key_id,
      s3_secret_access_key;
    std::optional<std::string> s3_region;
    if (!get_credentials(s3_endpoint,
                         bucket_name,
                         s3_access_key_id,
                         s3_secret_access_key,
                         s3_region)) {
        LOG_WARNING("Failed to get credentials. Skipping test.");
        return 0;
    }

    int retval = 1;
    const std::string object_name = "test-object";

    try {
        std::unique_ptr<zarr::S3Connection> conn;
        if (s3_region) {
            conn = std::make_unique<zarr::S3Connection>(
              s3_endpoint, s3_access_key_id, s3_secret_access_key, *s3_region);
        } else {
            conn = std::make_unique<zarr::S3Connection>(
              s3_endpoint, s3_access_key_id, s3_secret_access_key);
        }

        if (!conn->is_connection_valid()) {
            LOG_ERROR("Failed to connect to S3.");
            return 1;
        }
        CHECK(conn->bucket_exists(bucket_name));
        CHECK(conn->delete_object(bucket_name, object_name));
        CHECK(!conn->object_exists(bucket_name, object_name));

        std::vector<uint8_t> data(1024, 0);

        std::string etag =
          conn->put_object(bucket_name,
                           object_name,
                           std::span<uint8_t>(data.data(), data.size()));
        CHECK(!etag.empty());

        CHECK(conn->object_exists(bucket_name, object_name));

        // cleanup
        CHECK(conn->delete_object(bucket_name, object_name));

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed: ", e.what());
    }

    return retval;
}