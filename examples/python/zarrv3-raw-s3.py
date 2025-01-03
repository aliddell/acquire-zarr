# zarr_v3_s3_raw.py
import numpy as np
from acquire_zarr import (
    StreamSettings, ZarrStream, Dimension, DimensionType, ZarrVersion,
    DataType, S3Settings
)


def make_sample_data():
    return np.random.randint(
        0, 65535,
        (5, 2, 48, 64),  # Shape matches chunk sizes
        dtype=np.uint16
    )

def main():
    settings = StreamSettings()

    # Configure S3
    settings.s3 = S3Settings(
        endpoint="http://localhost:9000",
        bucket_name="mybucket",
        access_key_id="myaccesskey",
        secret_access_key="mysecretkey"
    )

    # Configure 4D array (t, z, y, x)
    settings.dimensions.extend([
        Dimension(
            name="t",
            kind=DimensionType.TIME,
            array_size_px=0,  # Unlimited
            chunk_size_px=5,
            shard_size_chunks=2,
        ),
        Dimension(
            name="z",
            kind=DimensionType.SPACE,
            array_size_px=10,
            chunk_size_px=2,
            shard_size_chunks=1,
        ),
        Dimension(
            name="y",
            kind=DimensionType.SPACE,
            array_size_px=48,
            chunk_size_px=16,
            shard_size_chunks=1,
        ),
        Dimension(
            name="x",
            kind=DimensionType.SPACE,
            array_size_px=64,
            chunk_size_px=16,
            shard_size_chunks=2,
        ),
    ])

    settings.store_path = "output_v3_s3.zarr"
    settings.version = ZarrVersion.V3
    settings.data_type = DataType.UINT16

    # Create stream
    stream = ZarrStream(settings)

    # Write sample data
    stream.append(make_sample_data())


if __name__ == "__main__":
    main()