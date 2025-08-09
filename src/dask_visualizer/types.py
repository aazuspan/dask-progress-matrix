from typing import Any, Unpack

ChunkIndex = tuple[Unpack[tuple[int, ...]], int, int]
"""The (..., x, y) indices of a chunk within a computation."""

IndexedTaskKey = tuple[str, Unpack[ChunkIndex]]
"""The task name and corresponding index of a Dask computation chunk."""

TaskKey = str | IndexedTaskKey
"""A generic Dask task key."""

Graph = dict[TaskKey, Any]
"""A lowered Dask graph mapping keys to computations."""

State = dict[str, Any]
