/// @file zarrv3-raw-s3.c
/// @brief Zarr V3 with uncompressed data to S3
#include "acquire.zarr.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Configure S3
    // Ensure that you have set your S3 credentials in the environment variables
    // AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and optionally AWS_SESSION_TOKEN
    ZarrS3Settings s3 = {
        .endpoint = "http://localhost:9000",
        .bucket_name = "mybucket",
    };

    // Configure stream settings
    ZarrArraySettings array = {
        .compression_settings = NULL, // No compression
        .data_type = ZarrDataType_uint16,
    };
    ZarrStreamSettings settings = {
        .store_path = "output_v3_s3.zarr",
        .s3_settings = &s3,
        .version = ZarrVersion_3,
        .max_threads = 0, // use all available threads
        .arrays = &array,
        .array_count = 1,
    };

    // Set up dimensions (t, z, y, x)
    ZarrArraySettings_create_dimension_array(settings.arrays, 4);

    settings.arrays->dimensions[0] = (ZarrDimensionProperties){
        .name = "t",
        .type = ZarrDimensionType_Time,
        .array_size_px = 0,  // Unlimited
        .chunk_size_px = 5,
        .shard_size_chunks = 2
    };

    settings.arrays->dimensions[1] = (ZarrDimensionProperties){
        .name = "z",
        .type = ZarrDimensionType_Space,
        .array_size_px = 10,
        .chunk_size_px = 2,
        .shard_size_chunks = 1
    };

    settings.arrays->dimensions[2] = (ZarrDimensionProperties){
        .name = "y",
        .type = ZarrDimensionType_Space,
        .array_size_px = 48,
        .chunk_size_px = 16,
        .shard_size_chunks = 1
    };

    settings.arrays->dimensions[3] = (ZarrDimensionProperties){
        .name = "x",
        .type = ZarrDimensionType_Space,
        .array_size_px = 64,
        .chunk_size_px = 16,
        .shard_size_chunks = 2
    };

    // Create stream
    ZarrStream* stream = ZarrStream_create(&settings);
    // Free Dimension array
    ZarrArraySettings_destroy_dimension_array(settings.arrays);

    if (!stream) {
        fprintf(stderr, "Failed to create stream\n");
        return 1;
    }

    // Create sample data
    const size_t width = 64;
    const size_t height = 48;
    uint16_t* frame = (uint16_t*)malloc(width * height * sizeof(uint16_t));

    // Write frames
    size_t bytes_written;
    for (int i = 0; i < 10; i++) {
        // Fill frame with sample data
        for (size_t j = 0; j < width * height; j++) {
            frame[j] = i * 1000 + j;
        }

        ZarrStatusCode status =
          ZarrStream_append(stream,
                            frame,
                            width * height * sizeof(uint16_t),
                            &bytes_written,
                            NULL);

        if (status != ZarrStatusCode_Success) {
            fprintf(stderr, "Failed to append frame: %s\n",
                    Zarr_get_status_message(status));
            break;
        }
    }

    // Cleanup
    free(frame);
    ZarrStream_destroy(stream);
    return 0;
}