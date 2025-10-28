"""Tests for distributed ProgressMatrix."""

import dask.array as da
import pytest
from distributed import Client, LocalCluster

from dask_progress_matrix.distributed import ProgressMatrix


@pytest.fixture
def distributed_client():
    """Create a local distributed client for testing."""
    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, processes=False, silence_logs=True
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def test_distributed_progress_matrix_basic(distributed_client):
    """Test that distributed ProgressMatrix works with a basic computation."""
    with ProgressMatrix(distributed_client, scale=2):
        x = da.random.random((64, 64), chunks=(32, 32))
        result = x.compute()

    assert result.shape == (64, 64)


def test_distributed_progress_matrix_multiple_arrays(distributed_client):
    """Test computing multiple arrays within a single context."""
    with ProgressMatrix(distributed_client, scale=2):
        x1 = da.random.random((64, 64), chunks=(32, 32))
        result1 = x1.compute()

        x2 = da.random.random((32, 32), chunks=(16, 16))
        result2 = x2.compute()

    assert result1.shape == (64, 64)
    assert result2.shape == (32, 32)


def test_distributed_progress_matrix_no_chunks(distributed_client):
    """Test that computations with no chunked tasks don't crash."""
    with ProgressMatrix(distributed_client):
        x = da.random.random((64,), chunks=(64,))
        result = x.compute()

    assert result.shape == (64,)


@pytest.mark.parametrize("mode", ["index", "elapsed"])
def test_distributed_progress_matrix_modes(distributed_client, mode):
    """Test different display modes."""
    with ProgressMatrix(distributed_client, mode=mode, scale=2):
        x = da.random.random((64, 64), chunks=(32, 32))
        result = x.compute()

    assert result.shape == (64, 64)


def test_distributed_progress_matrix_with_xarray(distributed_client):
    """Test that ProgressMatrix works with xarray objects."""
    import xarray as xr

    arr = da.random.random((64, 64), chunks=(32, 32))
    ds = xr.Dataset({"data": (("x", "y"), arr)})

    with ProgressMatrix(distributed_client, scale=2):
        result = ds.compute()

    assert result["data"].shape == (64, 64)
