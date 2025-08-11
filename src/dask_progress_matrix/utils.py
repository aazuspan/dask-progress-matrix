from __future__ import annotations

import random
import time
from typing import cast

import dask.array

from dask_progress_matrix.types import ChunkIndex, Graph, IndexedTaskKey


def generate_slow_dask_array(
    shape, chunks, randomize=True, delay: float = 1.0
) -> dask.array.Array:
    """
    Generate a lazily computed Dask array where each chunk's computation is delayed.

    This can be used to simulate real work within a chunked computation.
    """
    da = dask.array.zeros(shape, chunks=chunks)

    def delayed_compute(chunk):
        sleep_time = max(random.normalvariate(delay, delay), 0) if randomize else delay

        time.sleep(sleep_time)
        return chunk

    return dask.array.apply_gufunc(
        delayed_compute,
        "()->()",
        da,
    )


def get_terminal_tasks(dsk: Graph) -> set[IndexedTaskKey]:
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


def get_chunk_shape(indexes: list[ChunkIndex]) -> tuple[int, int]:
    """
    Count the height and width chunks in an output computation from its 2D indexes.
    """
    y = len(set([i[0] for i in indexes]))
    x = len(set([i[1] for i in indexes]))
    return y, x


def index_2d_from_key(key: IndexedTaskKey) -> ChunkIndex:
    """
    Parse a 2D index from a task key.

    1D keys, e.g. (name, x) will be expanded to (1, x).
    nD keyes, e.g. (name, ..., y, x) will be truncated to (y, x).
    """
    if len(key) < 2:
        raise ValueError(f"The key {key} must contain at least 1 dimension.")
    if len(key) == 2:
        return (0, key[1])
    return cast(tuple[int, int], (key[-2], key[-1]))
