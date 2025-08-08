from __future__ import annotations

import random
import time

import dask.array
from dask_visualizer.types import Graph, TaskKey


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


def get_terminal_tasks(dsk: Graph) -> set[TaskKey]:
    """
    Find the terminal tasks in a lowered Dask graph and return their keys.

    Terminal tasks are not depended on by other tasks in the graph, and represent the
    final output computations.
    """
    all_tasks = set([k for k in dsk if isinstance(k, tuple)])
    terminal_tasks = all_tasks.copy()

    for task in all_tasks:
        for dep in dsk[task].dependencies:
            if dep in terminal_tasks:
                terminal_tasks.remove(dep)

    return terminal_tasks


def get_chunk_shape(tasks: set[TaskKey]) -> tuple[int, int]:
    """
    Get the number of chunks in an output computation from a set of task keys.

    This assumes that there is at least one task for each chunk index, and their keys
    follow the format (name, ..., y, x).
    """
    ncols = len(set([t[-1] for t in tasks]))
    nrows = len(set([t[-2] for t in tasks]))
    return nrows, ncols
