from typing import Any

from typing_extensions import Unpack

ChunkIndex = tuple[int, int]
"""The 2d indices for a chunk in a computation."""

IndexedTaskKey = tuple[str, Unpack[tuple[int, Unpack[tuple[int, ...]]]]]
"""The task name and corresponding index of a Dask computation chunk."""

TaskKey = str | IndexedTaskKey
"""A generic Dask task key."""

Graph = dict[TaskKey, Any]
"""A lowered Dask graph mapping keys to computations."""

State = dict[str, Any]
