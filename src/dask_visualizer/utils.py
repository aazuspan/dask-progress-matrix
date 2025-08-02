from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

import dask.array

if TYPE_CHECKING:
    import xarray as xr


def generate_slow_dask_array(
    shape, chunks, randomize=True, delay: float = 1.0
) -> dask.array.Array:
    """
    Generate a lazily computed Dask array where each chunk's computation is delayed.
    """
    da = dask.array.zeros(shape, chunks=chunks)

    def delayed_compute(chunk):
        sleep_time = max(random.normalvariate(delay), 0) if randomize else delay

        time.sleep(sleep_time)
        return chunk

    return dask.array.apply_gufunc(
        delayed_compute,
        "()->()",
        da,
    )


def extract_dask_array(
    obj: xr.Dataset | xr.DataArray | dask.array.Array,
) -> dask.array.Array:
    """Extract a Dask Array from an object that may be wrapping it."""
    # xarray.Dataset
    if hasattr(obj, "to_dataarray"):
        obj = obj.to_dataarray()
    # xarray.DataArray
    if hasattr(obj, "data"):
        obj = obj.data
    if isinstance(obj, dask.array.Array):
        return obj

    raise ValueError(
        f"Failed to extract a Dask Array from type {obj.__class__.__name__}."
    )
