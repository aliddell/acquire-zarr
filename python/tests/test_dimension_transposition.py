import json
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

DIMS = [
    Dimension(
        name="t",
        kind=DimensionType.TIME,
        array_size_px=1,
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
        name="z",
        kind=DimensionType.SPACE,
        array_size_px=3,
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


@pytest.mark.parametrize(
    "input_dims,output_dims,expected_frame_values",
    [
        (["t", "c", "z", "y", "x"], None, [0, 1, 2, 3, 4, 5]),
        (["t", "c", "z", "y", "x"], ["t", "c", "z", "y", "x"], [0, 1, 2, 3, 4, 5]),
        (["t", "z", "c", "y", "x"], ["t", "c", "z", "y", "x"], [0, 2, 4, 1, 3, 5]),
    ],
)
def test_dimension_transposition(
    store_path: Path,
    input_dims: list[str],
    output_dims: list[str] | None,
    expected_frame_values: list[int],
):
    """
    Test that dimensions provided in T, Z, C, Y, X order are correctly
    transposed to T, C, Z, Y, X for storage and metadata when dimension_order
    is explicitly specified.
    """
    dims = [next(dim for dim in DIMS if dim.name == name) for name in input_dims]
    array = ArraySettings(
        dimensions=dims,
        dimension_order=output_dims,
    )

    settings = StreamSettings(store_path=str(store_path), arrays=[array])
    stream = ZarrStream(settings)

    shape = tuple(dim.array_size_px for dim in array.dimensions)
    n_frames = np.prod(shape[:-2])
    for i in range(n_frames):
        stream.append(np.full(shape[-2:], i, dtype=np.uint8))
    stream.close()

    # Verify metadata has axes in prescribed order
    group_metadata = json.loads(Path(store_path / "zarr.json").read_text())
    axes = group_metadata["attributes"]["ome"]["multiscales"][0]["axes"]

    # Check that axes are in expected order
    axis_names = [ax["name"] for ax in axes]
    expected_axis_names = output_dims or input_dims
    assert axis_names == expected_axis_names, (
        f"Expected axes in {expected_axis_names} order, got {axis_names}"
    )

    # Verify data is stored in prescribed order
    data = np.asarray(zarr.open_array(store_path / "0"))
    sizes = {dim.name: dim.array_size_px or 1 for dim in dims}
    if output_dims:
        expected_shape = tuple(sizes[name] for name in output_dims)
    else:
        expected_shape = tuple(sizes[name] for name in input_dims)
    assert data.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {data.shape}"
    )

    # Verify transposed frame order
    np.testing.assert_equal(data[..., 0, 0].ravel(), expected_frame_values)


def test_transpose_dimension_0_raises_error():
    """Test that transposing dimension 0 away raises an error."""
    dims = [next(dim for dim in DIMS if dim.name == name) for name in ["z", "c", "y", "x"]]
    with pytest.raises(TypeError, match="Transposing dimension 0.*not currently supported"):
        ArraySettings(dimensions=dims, dimension_order=["c", "z", "y", "x"])
