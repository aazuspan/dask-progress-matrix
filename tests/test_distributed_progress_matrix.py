"""Tests for distributed ProgressMatrix."""

import time

import dask.array
import pytest
import xarray as xr

from .conftest import CapturedDistributedProgressMatrix


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5], ids=lambda i: f"{i}d")
def test_distributed_progress_matrix_dimensionality(
    dimensions: int, distributed_client, svg_snapshot
):
    """Test that you can generate a progress matrix with arbitrary dimensionality."""
    shape = (2, 2, 2, 64, 64)[-dimensions:]
    chunks = tuple([d // 2 for d in shape])
    da = dask.array.zeros(shape, chunks=chunks)
    with CapturedDistributedProgressMatrix(distributed_client, scale=4) as captured:
        da.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


@pytest.mark.parametrize("cmap", ["viridis", "inferno"])
def test_distributed_progress_matrix_cmap(cmap: str, distributed_client, svg_snapshot):
    """Test that you can set the colormap of the progress matrix."""
    da = dask.array.zeros((64, 64), chunks=(32, 32))

    with CapturedDistributedProgressMatrix(
        distributed_client,
        cmap=cmap,
        mode="index",
        show_legend=True,
        scale=4,
    ) as captured:
        da.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


@pytest.mark.parametrize("show_legend", [True, False], ids=("legend", "nolegend"))
def test_distributed_progress_matrix_show_legend(
    show_legend: bool, distributed_client, svg_snapshot
):
    """Test that you can toggle the legend of the progress matrix."""
    da = dask.array.zeros((64, 64), chunks=(32, 32))

    with CapturedDistributedProgressMatrix(
        distributed_client, show_legend=show_legend
    ) as captured:
        da.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


@pytest.mark.parametrize("target_width", [4, 16, 32])
def test_distributed_progress_matrix_target_width(
    target_width: int, distributed_client, svg_snapshot
):
    """Test that you can set the target width of the progress matrix."""
    da = dask.array.zeros((64,), chunks=(32,))

    with CapturedDistributedProgressMatrix(
        distributed_client, target_width=target_width
    ) as captured:
        da.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_distributed_progress_matrix_scale(
    scale: int, distributed_client, svg_snapshot
):
    """Test that you can set the scale of the progress matrix."""
    da = dask.array.zeros((64,), chunks=(32,))

    with CapturedDistributedProgressMatrix(distributed_client, scale=scale) as captured:
        da.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


def test_distributed_progress_matrix_multiple_arrays(distributed_client, svg_snapshot):
    """Test that you can compute multiple objects within a single progress context."""
    da1 = dask.array.zeros((64, 64), chunks=(32, 32))
    da2 = dask.array.zeros((16, 64), chunks=(16, 16))

    with CapturedDistributedProgressMatrix(
        distributed_client, target_width=32
    ) as captured:
        da1.compute()
        time.sleep(0.3)  # Allow time for display to update
        da2.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


@pytest.mark.parametrize("as_dataarray", [False, True], ids=["dataset", "dataarray"])
def test_distributed_progress_matrix_xarray(
    as_dataarray: bool, distributed_client, svg_snapshot
):
    """Test that you can compute an xarray object within a progress context."""

    da = dask.array.zeros((64, 64), chunks=(32, 32))
    ds: xr.Dataset | xr.DataArray = xr.Dataset({"data": (("x", "y"), da)})
    if as_dataarray:
        ds = ds.to_dataarray()

    with CapturedDistributedProgressMatrix(
        distributed_client, target_width=32
    ) as captured:
        ds.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)


def test_distributed_progress_matrix_with_no_tasks(distributed_client, svg_snapshot):
    """Test that a computation with no terminal tasks is ignored without errors."""
    da = dask.array.zeros((64,), chunks=(64,))

    with CapturedDistributedProgressMatrix(distributed_client) as captured:
        da.compute()
        time.sleep(0.3)  # Allow time for display to update
        svg_snapshot(captured.svg)
