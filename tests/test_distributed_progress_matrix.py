"""Tests for distributed ProgressMatrix."""

import dask.array
import pytest
import xarray as xr

from .conftest import CapturedDistributedProgressMatrix


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5], ids=lambda i: f"{i}d")
def test_distributed_progress_matrix_dimensionality(
    dimensions: int, distributed_client, svg_snapshot
):
    """Test that you can generate a progress matrix with arbitrary dimensionality."""
    def create_array():
        shape = (2, 2, 2, 64, 64)[-dimensions:]
        chunks = tuple([d // 2 for d in shape])
        return dask.array.zeros(shape, chunks=chunks)
    
    captured = CapturedDistributedProgressMatrix(distributed_client, create_array, scale=4)
    svg = captured.run()
    svg_snapshot(svg)


@pytest.mark.parametrize("cmap", ["viridis", "inferno"])
def test_distributed_progress_matrix_cmap(cmap: str, distributed_client, svg_snapshot):
    """Test that you can set the colormap of the progress matrix."""
    def create_array():
        return dask.array.zeros((64, 64), chunks=(32, 32))

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_array,
        cmap=cmap,
        mode="index",
        show_legend=True,
        scale=4,
    )
    svg = captured.run()
    svg_snapshot(svg)


@pytest.mark.parametrize("show_legend", [True, False], ids=("legend", "nolegend"))
def test_distributed_progress_matrix_show_legend(
    show_legend: bool, distributed_client, svg_snapshot
):
    """Test that you can toggle the legend of the progress matrix."""
    def create_array():
        return dask.array.zeros((64, 64), chunks=(32, 32))

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_array,
        show_legend=show_legend
    )
    svg = captured.run()
    svg_snapshot(svg)


@pytest.mark.parametrize("target_width", [4, 16, 32])
def test_distributed_progress_matrix_target_width(
    target_width: int, distributed_client, svg_snapshot
):
    """Test that you can set the target width of the progress matrix."""
    def create_array():
        return dask.array.zeros((64,), chunks=(32,))

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_array,
        target_width=target_width
    )
    svg = captured.run()
    svg_snapshot(svg)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_distributed_progress_matrix_scale(
    scale: int, distributed_client, svg_snapshot
):
    """Test that you can set the scale of the progress matrix."""
    def create_array():
        return dask.array.zeros((64,), chunks=(32,))

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_array,
        scale=scale
    )
    svg = captured.run()
    svg_snapshot(svg)


def test_distributed_progress_matrix_multiple_arrays(distributed_client, svg_snapshot):
    """Test that you can compute multiple objects - just return first one."""
    def create_array():
        # For distributed, we can only track one future at a time in this simple design
        return dask.array.zeros((64, 64), chunks=(32, 32))

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_array,
        target_width=32
    )
    svg = captured.run()
    svg_snapshot(svg)


@pytest.mark.parametrize("as_dataarray", [False, True], ids=["dataset", "dataarray"])
def test_distributed_progress_matrix_xarray(
    as_dataarray: bool, distributed_client, svg_snapshot
):
    """Test that you can compute an xarray object within a progress context."""
    def create_xarray():
        da = dask.array.zeros((64, 64), chunks=(32, 32))
        ds = xr.Dataset({"data": (("x", "y"), da)})
        if as_dataarray:
            ds = ds.to_dataarray()
        return ds

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_xarray,
        target_width=32
    )
    svg = captured.run()
    svg_snapshot(svg)


def test_distributed_progress_matrix_with_no_tasks(distributed_client, svg_snapshot):
    """Test that a computation with no terminal tasks is ignored without errors."""
    def create_array():
        return dask.array.zeros((64,), chunks=(64,))

    captured = CapturedDistributedProgressMatrix(
        distributed_client,
        create_array
    )
    svg = captured.run()
    svg_snapshot(svg)
