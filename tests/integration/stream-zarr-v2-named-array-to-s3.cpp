#include "acquire.zarr.h"
#include "test.macros.hh"

#include <nlohmann/json.hpp>
#include <miniocpp/client.h>

#include <vector>

namespace {
std::string s3_endpoint, s3_bucket_name, s3_access_key_id, s3_secret_access_key,
  s3_region;

const unsigned int array_width = 64, array_height = 48, array_planes = 6,
                   array_channels = 8, array_timepoints = 10;

const unsigned int chunk_width = 16, chunk_height = 16, chunk_planes = 2,
                   chunk_channels = 4, chunk_timepoints = 5;

const unsigned int chunks_in_x =
  (array_width + chunk_width - 1) / chunk_width; // 4 chunks
const unsigned int chunks_in_y =
  (array_height + chunk_height - 1) / chunk_height; // 3 chunks
const unsigned int chunks_in_z =
  (array_planes + chunk_planes - 1) / chunk_planes; // 3 chunks
const unsigned int chunks_in_c =
  (array_channels + chunk_channels - 1) / chunk_channels; // 2 chunks
const unsigned int chunks_in_t =
  (array_timepoints + chunk_timepoints - 1) / chunk_timepoints;

const size_t nbytes_px = sizeof(int32_t);
const uint32_t frames_to_acquire =
  array_planes * array_channels * array_timepoints;
const size_t bytes_of_frame = array_width * array_height * nbytes_px;

bool
get_credentials()
{
    char* env = nullptr;
    if (!(env = std::getenv("ZARR_S3_ENDPOINT"))) {
        LOG_ERROR("ZARR_S3_ENDPOINT not set.");
        return false;
    }
    s3_endpoint = env;

    if (!(env = std::getenv("ZARR_S3_BUCKET_NAME"))) {
        LOG_ERROR("ZARR_S3_BUCKET_NAME not set.");
        return false;
    }
    s3_bucket_name = env;

    if (!(env = std::getenv("AWS_ACCESS_KEY_ID"))) {
        LOG_ERROR("AWS_ACCESS_KEY_ID not set.");
        return false;
    }
    s3_access_key_id = env;

    if (!(env = std::getenv("AWS_SECRET_ACCESS_KEY"))) {
        LOG_ERROR("AWS_SECRET_ACCESS_KEY not set.");
        return false;
    }
    s3_secret_access_key = env;

    env = std::getenv("ZARR_S3_REGION");
    if (env) {
        s3_region = env;
    }

    return true;
}

bool
object_exists(minio::s3::Client& client, const std::string& object_name)
{
    minio::s3::StatObjectArgs args;
    args.bucket = s3_bucket_name;
    args.object = object_name;

    minio::s3::StatObjectResponse response = client.StatObject(args);

    return (bool)response;
}

size_t
get_object_size(minio::s3::Client& client, const std::string& object_name)
{
    minio::s3::StatObjectArgs args;
    args.bucket = s3_bucket_name;
    args.object = object_name;

    minio::s3::StatObjectResponse response = client.StatObject(args);

    if (!response) {
        LOG_ERROR("Failed to get object size: %s", object_name.c_str());
        return 0;
    }

    return response.size;
}

std::string
get_object_contents(minio::s3::Client& client, const std::string& object_name)
{
    std::stringstream ss;

    minio::s3::GetObjectArgs args;
    args.bucket = s3_bucket_name;
    args.object = object_name;
    args.datafunc = [&ss](minio::http::DataFunctionArgs args) -> bool {
        ss << args.datachunk;
        return true;
    };

    // Call get object.
    minio::s3::GetObjectResponse resp = client.GetObject(args);

    return ss.str();
}

bool
remove_items(minio::s3::Client& client,
             const std::vector<std::string>& item_keys)
{
    std::list<minio::s3::DeleteObject> objects;
    for (const auto& key : item_keys) {
        minio::s3::DeleteObject object;
        object.name = key;
        objects.push_back(object);
    }

    minio::s3::RemoveObjectsArgs args;
    args.bucket = s3_bucket_name;

    auto it = objects.begin();

    args.func = [&objects = objects,
                 &i = it](minio::s3::DeleteObject& obj) -> bool {
        if (i == objects.end())
            return false;
        obj = *i;
        i++;
        return true;
    };

    minio::s3::RemoveObjectsResult result = client.RemoveObjects(args);
    for (; result; result++) {
        minio::s3::DeleteError err = *result;
        if (!err) {
            LOG_ERROR("Failed to delete object %s: %s",
                      err.object_name.c_str(),
                      err.message.c_str());
            return false;
        }
    }

    return true;
}
} // namespace

ZarrStream*
setup()
{
    ZarrArraySettings array = {
        .output_key = "intermediate/path",
        .compression_settings = nullptr,
        .data_type = ZarrDataType_int32,
    };
    ZarrStreamSettings settings = {
        .store_path = TEST,
        .s3_settings = nullptr,
        .version = ZarrVersion_2,
        .max_threads = 0, // use all available threads
        .arrays = &array,
        .array_count = 1,
    };

    ZarrS3Settings s3_settings{
        .endpoint = s3_endpoint.c_str(),
        .bucket_name = s3_bucket_name.c_str(),
    };
    if (!s3_region.empty()) {
        s3_settings.region = s3_region.c_str();
    }

    settings.s3_settings = &s3_settings;

    CHECK_OK(ZarrArraySettings_create_dimension_array(settings.arrays, 5));

    ZarrDimensionProperties* dim;
    dim = settings.arrays->dimensions;
    *dim = DIM("t",
               ZarrDimensionType_Time,
               array_timepoints,
               chunk_timepoints,
               0,
               nullptr,
               1.0);

    dim = settings.arrays->dimensions + 1;
    *dim = DIM("c",
               ZarrDimensionType_Channel,
               array_channels,
               chunk_channels,
               0,
               nullptr,
               1.0);

    dim = settings.arrays->dimensions + 2;
    *dim = DIM("z",
               ZarrDimensionType_Space,
               array_planes,
               chunk_planes,
               0,
               "millimeter",
               1.4);

    dim = settings.arrays->dimensions + 3;
    *dim = DIM("y",
               ZarrDimensionType_Space,
               array_height,
               chunk_height,
               0,
               "micrometer",
               0.9);

    dim = settings.arrays->dimensions + 4;
    *dim = DIM("x",
               ZarrDimensionType_Space,
               array_width,
               chunk_width,
               0,
               "micrometer",
               0.9);

    auto* stream = ZarrStream_create(&settings);
    ZarrArraySettings_destroy_dimension_array(settings.arrays);

    return stream;
}

void
verify_group_metadata(const nlohmann::json& meta)
{
    const auto zarr_format = meta["zarr_format"].get<int>();
    EXPECT_EQ(int, zarr_format, 2);
}

void
verify_array_metadata(const nlohmann::json& meta)
{
    const auto& shape = meta["shape"];
    EXPECT_EQ(size_t, shape.size(), 5);
    EXPECT_EQ(int, shape[0].get<int>(), array_timepoints);
    EXPECT_EQ(int, shape[1].get<int>(), array_channels);
    EXPECT_EQ(int, shape[2].get<int>(), array_planes);
    EXPECT_EQ(int, shape[3].get<int>(), array_height);
    EXPECT_EQ(int, shape[4].get<int>(), array_width);

    const auto& chunks = meta["chunks"];
    EXPECT_EQ(size_t, chunks.size(), 5);
    EXPECT_EQ(int, chunks[0].get<int>(), chunk_timepoints);
    EXPECT_EQ(int, chunks[1].get<int>(), chunk_channels);
    EXPECT_EQ(int, chunks[2].get<int>(), chunk_planes);
    EXPECT_EQ(int, chunks[3].get<int>(), chunk_height);
    EXPECT_EQ(int, chunks[4].get<int>(), chunk_width);

    const auto dtype = meta["dtype"].get<std::string>();
    EXPECT(dtype == "<i4",
           "Expected dtype to be '<i4', but got '%s'",
           dtype.c_str());

    const auto& compressor = meta["compressor"];
    EXPECT(compressor.is_null(),
           "Expected compressor to be null, but got '%s'",
           compressor.dump().c_str());
}

void
verify_and_cleanup()
{

    minio::s3::BaseUrl url(s3_endpoint);
    url.https = s3_endpoint.starts_with("https://");

    minio::creds::StaticProvider provider(s3_access_key_id,
                                          s3_secret_access_key);
    minio::s3::Client client(url, &provider);

    std::string base_metadata_path = TEST "/.zgroup";
    std::string group_metadata_path = TEST "/intermediate/.zgroup";
    std::string array_metadata_path = TEST "/intermediate/path/.zarray";

    {
        EXPECT(object_exists(client, base_metadata_path),
               "Object does not exist: %s",
               base_metadata_path.c_str());
        std::string contents = get_object_contents(client, base_metadata_path);
        nlohmann::json group_metadata = nlohmann::json::parse(contents);

        verify_group_metadata(group_metadata);
    }

    {
        EXPECT(object_exists(client, group_metadata_path),
               "Object does not exist: %s",
               group_metadata_path.c_str());
        std::string contents = get_object_contents(client, group_metadata_path);
        nlohmann::json group_metadata = nlohmann::json::parse(contents);

        verify_group_metadata(group_metadata);
    }

    {
        EXPECT(object_exists(client, array_metadata_path),
               "Object does not exist: %s",
               array_metadata_path.c_str());
        std::string contents = get_object_contents(client, array_metadata_path);
        nlohmann::json array_metadata = nlohmann::json::parse(contents);

        verify_array_metadata(array_metadata);
    }

    CHECK(remove_items(
      client,
      { base_metadata_path, group_metadata_path, array_metadata_path }));

    const auto expected_file_size = chunk_width * chunk_height * chunk_planes *
                                    chunk_channels * chunk_timepoints *
                                    nbytes_px;

    // verify and clean up data files
    std::vector<std::string> data_files;
    std::string data_root = TEST "/intermediate/path";

    for (auto t = 0; t < chunks_in_t; ++t) {
        const auto t_dir = data_root + "/" + std::to_string(t);

        for (auto c = 0; c < chunks_in_c; ++c) {
            const auto c_dir = t_dir + "/" + std::to_string(c);

            for (auto z = 0; z < chunks_in_z; ++z) {
                const auto z_dir = c_dir + "/" + std::to_string(z);

                for (auto y = 0; y < chunks_in_y; ++y) {
                    const auto y_dir = z_dir + "/" + std::to_string(y);

                    for (auto x = 0; x < chunks_in_x; ++x) {
                        const auto x_file = y_dir + "/" + std::to_string(x);
                        EXPECT(object_exists(client, x_file),
                               "Object does not exist: ",
                               x_file);
                        const auto file_size = get_object_size(client, x_file);
                        EXPECT_EQ(size_t, file_size, expected_file_size);
                        data_files.push_back(x_file);
                    }

                    CHECK(!object_exists(
                      client, y_dir + "/" + std::to_string(chunks_in_x)));
                }
            }
        }
    }

    CHECK(remove_items(client, data_files));
}

int
main()
{
    if (!get_credentials()) {
        LOG_WARNING("Failed to get credentials. Skipping test.");
        return 0;
    }

    Zarr_set_log_level(ZarrLogLevel_Debug);

    auto* stream = setup();
    std::vector<int32_t> frame(array_width * array_height, 0);

    int retval = 1;

    try {
        size_t bytes_out;
        for (auto i = 0; i < frames_to_acquire; ++i) {
            ZarrStatusCode status = ZarrStream_append(
              stream, frame.data(), bytes_of_frame, &bytes_out, nullptr);
            EXPECT(status == ZarrStatusCode_Success,
                   "Failed to append frame ",
                   i,
                   ": ",
                   Zarr_get_status_message(status));
            EXPECT_EQ(size_t, bytes_out, bytes_of_frame);
        }

        ZarrStream_destroy(stream);

        verify_and_cleanup();

        retval = 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Caught exception: %s", e.what());
    }

    return retval;
}
