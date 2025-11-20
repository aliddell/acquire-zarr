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

DIMS = {
    "t": Dimension(
        name="t",
        kind=DimensionType.TIME,
        array_size_px=2,
        chunk_size_px=1,
        shard_size_chunks=1,
    ),
    "c": Dimension(
        name="c",
        kind=DimensionType.CHANNEL,
        array_size_px=3,
        chunk_size_px=1,
        shard_size_chunks=1,
    ),
    "z": Dimension(
        name="z",
        kind=DimensionType.SPACE,
        array_size_px=4,
        chunk_size_px=1,
        shard_size_chunks=1,
    ),
    "y": Dimension(
        name="y",
        kind=DimensionType.SPACE,
        array_size_px=16,
        chunk_size_px=8,
        shard_size_chunks=1,
    ),
    "x": Dimension(
        name="x",
        kind=DimensionType.SPACE,
        array_size_px=24,
        chunk_size_px=8,
        shard_size_chunks=1,
    ),
}


@pytest.mark.parametrize(
    "input_dims,output_dims",
    [
        (["t", "c", "z", "y", "x"], None),
        (["t", "c", "z", "y", "x"], ["t", "c", "z", "y", "x"]),
        (["t", "z", "c", "y", "x"], ["t", "c", "z", "y", "x"]),
    ],
)
def test_dimension_transposition(
    store_path: Path, input_dims: list[str], output_dims: list[str] | None
):
    """
    Test that data received in `input_dims` order is correctly stored
    according to the specified `output_dims` order.

    Frames are written sequentially (frame 0, 1, 2, ...) where each frame
    corresponds to iterating through the append dimensions in input_dims order.
    The test verifies that these frames end up in the correct positions when
    stored according to output_dims order.
    """
    array = ArraySettings(
        dimensions=[DIMS[name] for name in input_dims],
        dimension_order=output_dims,
    )
    settings = StreamSettings(store_path=str(store_path), arrays=[array])
    stream = ZarrStream(settings)

    output_dims = input_dims if output_dims is None else output_dims
    input_shape = tuple(DIMS[n].array_size_px for n in input_dims)
    output_shape = tuple(DIMS[n].array_size_px for n in output_dims)
    n_frames = np.prod(input_shape[:-2])
    if output_dims and output_dims != input_dims:
        assert input_shape != output_shape, (
            "Input and output shapes should differ for this test case"
        )

    # Write frames with sequential values (0, 1, 2, ...)
    # Frames are written in input dimension order
    expected_frame_values = np.arange(n_frames, dtype=np.uint8)
    for val in expected_frame_values:
        stream.append(np.full(input_shape[-2:], val, dtype=np.uint8))
    stream.close()

    # Verify metadata has axes in prescribed order
    group_metadata = json.loads(Path(store_path / "zarr.json").read_text())
    axes = group_metadata["attributes"]["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]
    assert axis_names == output_dims, (
        f"Expected metadata axes in {output_dims} order, got {axis_names}"
    )

    # Verify data is stored in prescribed order
    written_data = np.asarray(zarr.open_array(store_path / "0"))
    assert written_data.shape == output_shape, (
        f"Expected written data with shape {output_shape}, got {written_data.shape}"
    )

    # Each frame was written with np.full(), so all pixels have the same value.
    # Extract one value per plane to get the frame numbers as stored.
    stored_frame_values = written_data[..., 0, 0]

    # Build expected frame values: start in input order, transpose if needed
    # we need to reshape because expected_frame_values is 1D initially but
    # stored_frame_values is in the full output shape
    expected_frame_values = expected_frame_values.reshape(input_shape[:-2])
    if output_dims and output_dims != input_dims:
        perm = [input_dims.index(d) for d in output_dims[:-2]]
        expected_frame_values = np.transpose(expected_frame_values, perm)

    # Verify the stored frame values match the expected transposition
    np.testing.assert_array_equal(stored_frame_values, expected_frame_values)


def test_transpose_dimension_0_raises_error():
    """Test that transposing dimension 0 away raises an error."""
    dims = [DIMS[name] for name in ["z", "c", "y", "x"]]
    with pytest.raises(
        TypeError, match="Transposing dimension 0.*not currently supported"
    ):
        ArraySettings(dimensions=dims, dimension_order=["c", "z", "y", "x"])
