#!/usr/bin/env python3
"""
Test that acquire-zarr correctly transposes dimensions to OME-NGFF order.

This test validates that when dimensions are provided in non-canonical order
(e.g., T, Z, C, Y, X), the library automatically:
1. Stores data in canonical TCZYX order
2. Writes metadata with axes in canonical TCZYX order
3. Transposes data correctly at write time
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from acquire_zarr import (
    ArraySettings,
    Dimension,
    DimensionType,
    StreamSettings,
    ZarrStream,
)


@pytest.fixture(scope="function")
def store_path(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_dimension_transposition_tzc_to_tcz(store_path):
    """
    Test that dimensions provided in T, Z, C, Y, X order are correctly
    transposed to T, C, Z, Y, X for storage and metadata when dimension_order
    is explicitly specified.
    """
    # User provides dimensions in acquisition order: T, Z, C, Y, X
    # (time varies slowest, then Z, then C, then Y, then X varies fastest)
    # and OPTS IN to transposition by specifying dimension_order
    settings = StreamSettings()
    settings.store_path = str(store_path)
    settings.arrays = [
        ArraySettings(
            dimensions=[
                Dimension(
                    name="t",
                    kind=DimensionType.TIME,
                    array_size_px=0,  # append dimension
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="z",
                    kind=DimensionType.SPACE,
                    array_size_px=3,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="c",
                    kind=DimensionType.CHANNEL,
                    array_size_px=2,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="y",
                    kind=DimensionType.SPACE,
                    array_size_px=4,
                    chunk_size_px=4,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="x",
                    kind=DimensionType.SPACE,
                    array_size_px=4,
                    chunk_size_px=4,
                    shard_size_chunks=1,
                ),
            ],
            # OPT-IN to transposition by specifying target dimension order
            dimension_order=["t", "c", "z", "y", "x"]
        )
    ]

    # Write frames in acquisition order: T, Z, C
    # Frame 0: t=0, z=0, c=0
    # Frame 1: t=0, z=0, c=1
    # Frame 2: t=0, z=1, c=0
    # Frame 3: t=0, z=1, c=1
    # Frame 4: t=0, z=2, c=0
    # Frame 5: t=0, z=2, c=1
    n_frames = 6
    frame_height, frame_width = 4, 4

    stream = ZarrStream(settings)
    for i in range(n_frames):
        # Create frame with unique value for each position
        frame = np.full((frame_height, frame_width), i, dtype=np.uint8)
        stream.append(frame)
    stream.close()
    # Verify metadata has axes in canonical TCZYX order
    with open(store_path / "zarr.json", "r") as f:
        group_metadata = json.load(f)

    axes = group_metadata["attributes"]["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]
    axis_types = [ax["type"] for ax in axes]

    # Check that axes are in canonical order: T, C, Z, Y, X
    assert axis_names == ["t", "c", "z", "y", "x"], (
        f"Expected axes in TCZYX order, got {axis_names}"
    )
    assert axis_types == ["time", "channel", "space", "space", "space"], (
        f"Expected types [time, channel, space, space, space], got {axis_types}"
    )

    # Verify data is stored in canonical TCZYX order
    root = zarr.open(store_path / "0", mode="r")
    data = np.array(root[:])

    # Data shape should be (1, 2, 3, 4, 4) = (T, C, Z, Y, X)
    assert data.shape == (1, 2, 3, 4, 4), (
        f"Expected shape (1, 2, 3, 4, 4), got {data.shape}"
    )

    # Verify data is correctly transposed from acquisition order to storage order
    # Acquisition order: T, Z, C, Y, X
    # Frame 0: t=0, z=0, c=0 -> should be at storage [t=0, c=0, z=0]
    # Frame 1: t=0, z=0, c=1 -> should be at storage [t=0, c=1, z=0]
    # Frame 2: t=0, z=1, c=0 -> should be at storage [t=0, c=0, z=1]
    # Frame 3: t=0, z=1, c=1 -> should be at storage [t=0, c=1, z=1]
    # Frame 4: t=0, z=2, c=0 -> should be at storage [t=0, c=0, z=2]
    # Frame 5: t=0, z=2, c=1 -> should be at storage [t=0, c=1, z=2]

    assert np.all(data[0, 0, 0, :, :] == 0), "Frame 0 data mismatch"
    assert np.all(data[0, 1, 0, :, :] == 1), "Frame 1 data mismatch"
    assert np.all(data[0, 0, 1, :, :] == 2), "Frame 2 data mismatch"
    assert np.all(data[0, 1, 1, :, :] == 3), "Frame 3 data mismatch"
    assert np.all(data[0, 0, 2, :, :] == 4), "Frame 4 data mismatch"
    assert np.all(data[0, 1, 2, :, :] == 5), "Frame 5 data mismatch"

    print("✓ Dimension transposition test passed!")
    print(f"  - Metadata axes are in canonical order: {axis_names}")
    print(f"  - Data shape is correct: {data.shape}")
    print("  - Data is correctly transposed from T,Z,C to T,C,Z")


def test_dimension_no_transposition_needed(store_path):
    """
    Test that when dimensions are already in canonical order,
    no transposition occurs and data is written correctly.
    """
    # User provides dimensions already in canonical order: T, C, Y, X
    settings = StreamSettings()
    settings.store_path = str(store_path)
    settings.arrays = [
        ArraySettings(
            dimensions=[
                Dimension(
                    name="t",
                    kind=DimensionType.TIME,
                    array_size_px=0,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="c",
                    kind=DimensionType.CHANNEL,
                    array_size_px=2,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="y",
                    kind=DimensionType.SPACE,
                    array_size_px=4,
                    chunk_size_px=4,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="x",
                    kind=DimensionType.SPACE,
                    array_size_px=4,
                    chunk_size_px=4,
                    shard_size_chunks=1,
                ),
            ]
        )
    ]

    n_frames = 4
    frame_height, frame_width = 4, 4

    stream = ZarrStream(settings)
    for i in range(n_frames):
        frame = np.full((frame_height, frame_width), i, dtype=np.uint8)
        stream.append(frame)
    stream.close()
    # Verify metadata
    with open(store_path / "zarr.json", "r") as f:
        group_metadata = json.load(f)

    axes = group_metadata["attributes"]["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]

    assert axis_names == ["t", "c", "y", "x"], (
        f"Expected axes [t, c, y, x], got {axis_names}"
    )

    # Verify data
    root = zarr.open(store_path / "0", mode="r")
    data = np.array(root[:])

    assert data.shape == (2, 2, 4, 4), f"Expected shape (2, 2, 4, 4), got {data.shape}"

    # Frames should be in order: t=0,c=0 -> t=0,c=1 -> t=1,c=0 -> t=1,c=1
    assert np.all(data[0, 0, :, :] == 0)
    assert np.all(data[0, 1, :, :] == 1)
    assert np.all(data[1, 0, :, :] == 2)
    assert np.all(data[1, 1, :, :] == 3)

    print("✓ No transposition test passed!")


def test_complex_dimension_order_tzcyx_to_tczyx(store_path):
    """
    Test more complex transposition: T, Z, C, Y, X -> T, C, Z, Y, X
    with multiple values in each dimension.
    """
    settings = StreamSettings()
    settings.store_path = str(store_path)
    settings.arrays = [
        ArraySettings(
            dimensions=[
                Dimension(
                    name="time",
                    kind=DimensionType.TIME,
                    array_size_px=0,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="depth",
                    kind=DimensionType.SPACE,
                    array_size_px=2,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="channel",
                    kind=DimensionType.CHANNEL,
                    array_size_px=3,
                    chunk_size_px=1,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="height",
                    kind=DimensionType.SPACE,
                    array_size_px=2,
                    chunk_size_px=2,
                    shard_size_chunks=1,
                ),
                Dimension(
                    name="width",
                    kind=DimensionType.SPACE,
                    array_size_px=2,
                    chunk_size_px=2,
                    shard_size_chunks=1,
                ),
            ],
            # OPT-IN to transposition by specifying target order
            dimension_order=["time", "channel", "depth", "height", "width"]
        )
    ]

    # 2 timepoints × 2 z-slices × 3 channels = 12 frames
    n_frames = 12
    frame_height, frame_width = 2, 2

    stream = ZarrStream(settings)
    for i in range(n_frames):
        frame = np.full((frame_height, frame_width), i, dtype=np.uint8)
        stream.append(frame)
    stream.close()
    
    # Verify axes order
    with open(store_path / "zarr.json", "r") as f:
        group_metadata = json.load(f)

    axes = group_metadata["attributes"]["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]

    # Should be reordered to T, C, Z, Y, X
    assert axis_names == ["time", "channel", "depth", "height", "width"], (
        f"Expected canonical order, got {axis_names}"
    )

    # Verify data shape and content
    root = zarr.open(store_path / "0", mode="r")
    data = np.array(root[:])

    # Shape should be (T=2, C=3, Z=2, Y=2, X=2)
    assert data.shape == (2, 3, 2, 2, 2), f"Expected shape (2,3,2,2,2), got {data.shape}"

    # Verify transposition: acquisition order is T,Z,C
    # Frame mapping:
    # acq frame  t z c  ->  storage [t,c,z]
    #     0      0 0 0  ->  [0,0,0]
    #     1      0 0 1  ->  [0,1,0]
    #     2      0 0 2  ->  [0,2,0]
    #     3      0 1 0  ->  [0,0,1]
    #     4      0 1 1  ->  [0,1,1]
    #     5      0 1 2  ->  [0,2,1]
    #     6      1 0 0  ->  [1,0,0]
    #     7      1 0 1  ->  [1,1,0]
    #     8      1 0 2  ->  [1,2,0]
    #     9      1 1 0  ->  [1,0,1]
    #    10      1 1 1  ->  [1,1,1]
    #    11      1 1 2  ->  [1,2,1]

    assert np.all(data[0, 0, 0, :, :] == 0), "Frame 0 position incorrect"
    assert np.all(data[0, 1, 0, :, :] == 1), "Frame 1 position incorrect"
    assert np.all(data[0, 2, 0, :, :] == 2), "Frame 2 position incorrect"
    assert np.all(data[0, 0, 1, :, :] == 3), "Frame 3 position incorrect"
    assert np.all(data[0, 1, 1, :, :] == 4), "Frame 4 position incorrect"
    assert np.all(data[0, 2, 1, :, :] == 5), "Frame 5 position incorrect"
    assert np.all(data[1, 0, 0, :, :] == 6), "Frame 6 position incorrect"
    assert np.all(data[1, 1, 0, :, :] == 7), "Frame 7 position incorrect"
    assert np.all(data[1, 2, 0, :, :] == 8), "Frame 8 position incorrect"
    assert np.all(data[1, 0, 1, :, :] == 9), "Frame 9 position incorrect"
    assert np.all(data[1, 1, 1, :, :] == 10), "Frame 10 position incorrect"
    assert np.all(data[1, 2, 1, :, :] == 11), "Frame 11 position incorrect"

    print("✓ Complex dimension transposition test passed!")
    print(f"  - Correctly transposed T,Z,C -> T,C,Z for {n_frames} frames")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        print("\nRunning dimension transposition tests...\n")
        test_dimension_transposition_tzc_to_tcz(tmp_path / "test1")
        print()
        test_dimension_no_transposition_needed(tmp_path / "test2")
        print()
        test_complex_dimension_order_tzcyx_to_tczyx(tmp_path / "test3")
        print("\n✅ All tests passed!")
