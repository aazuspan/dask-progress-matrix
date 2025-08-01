import random
import time

import dask.array


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
