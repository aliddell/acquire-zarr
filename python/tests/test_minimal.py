import acquire_zarr

import time

# time.sleep(10)  # Sleep for 10 seconds to allow the debugger to attach

# dim = acquire_zarr.Dimension(name="t", kind=acquire_zarr.DimensionType.TIME, array_size_px=2, chunk_size_px=1, shard_size_chunks=1)
c = acquire_zarr.CompressionSettings(
    # codec=acquire_zarr.CompressionCodec.BLOSC_LZ4,
    # compressor=acquire_zarr.Compressor.BLOSC1,
    # level=0,
    # shuffle=1
    None, None, 1, 1
)

def test_dummy():
    pass