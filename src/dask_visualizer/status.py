from collections import Counter
from dataclasses import dataclass
from enum import Enum

import dask.array
import numpy as np
from dask.layers import ArraySliceDep
from dask_visualizer.types import Graph, TaskKey
from numpy.typing import NDArray


class ComputationState(Enum):
    WAITING = 0
    STARTED = 1
    COMPLETE = 2


@dataclass
class ComputationChunk:
    # The (x, y) index of the chunk
    index: tuple[int, int]

    # The number of tasks remaining to complete the chunk. For a 2D array, this will be
    # initialized to 1 and decremented by 0.5 when the chunk is started and again when
    # it is finished. For a 3D array, there will be one task per coordinate in the 1st
    # dimension.
    tasks_remaining: int
    state: ComputationState = ComputationState.WAITING
    completed_idx: int | None = None

    def start(self):
        self.state = ComputationState.STARTED
        self.tasks_remaining -= 0.5

    def finish(self, idx: int):
        self.tasks_remaining -= 0.5
        if self.tasks_remaining == 0:
            self.completed_idx = idx
            self.state = ComputationState.COMPLETE


class ComputationStatus:
    """
    Track the computation state of a Dask array in a Numpy array.
    """

    def __init__(self, obj: dask.array.Array):
        self._obj = obj

        # Track the sequential index of the last completed slice
        self._current_idx = 0

        # An indexer from (x, y) chunk indexes to array indexes
        self._chunk_indexer = ArraySliceDep(obj.chunks[-2:])

        # A mapping from (x, y) chunk indices to computation chunks
        self._chunks: dict[tuple[int, int], ComputationChunk] = {}

    def _assert_initialized(self):
        """Ensure that the computation has been initialized."""
        if self.graph is None or self.state is None:
            raise AttributeError("The computation has not been initialized.")

    def _is_tracked_task(self, key: TaskKey) -> bool:
        """
        Check whether the given task should be tracked.

        This filters out intermediate tasks.
        """
        return isinstance(key, tuple) and key[0] == self._obj.name

    def initialize(self, dsk: Graph):
        """Triggered by the start of a computation."""
        self.graph = dsk

        # TODO: If the computation doesn't match the passed array, there will be no task
        # chunks. Handle that.
        chunk_indexes = [tuple(k[-2:]) for k in dsk if self._is_tracked_task(k)]
        for chunk_index, num_tasks in Counter(chunk_indexes).items():
            self._chunks[chunk_index] = ComputationChunk(chunk_index, num_tasks)

    @property
    def state(self) -> NDArray:
        # Once computation is complete, return the completion index of each chunk,
        # scaled [0, 2] to match the progress array.
        if all(
            [
                chunk.state is ComputationState.COMPLETE
                for chunk in self._chunks.values()
            ]
        ):
            return self._generate_index_array()

        return self._generate_progress_array()

    def _generate_progress_array(self) -> NDArray:
        # TODO: Generate this only at the necessary resolution
        state = np.zeros(self._obj.shape[-2:])
        for chunk_index, chunk in self._chunks.items():
            array_idx = self._chunk_indexer[chunk_index]
            state[array_idx] = chunk.state.value

        return state

    def _generate_index_array(self) -> NDArray:
        completion_indexes = np.array(
            [chunk.completed_idx for chunk in self._chunks.values()]
        )
        min_idx = completion_indexes.min()
        max_idx = (completion_indexes - min_idx).max()

        index = np.zeros(self._obj.shape[-2:])
        for chunk_index, chunk in self._chunks.items():
            array_idx = self._chunk_indexer[chunk_index]
            # Scale the index [0, 2] to match the task progress states
            index[array_idx] = (chunk.completed_idx - min_idx) / (max_idx * 0.5)

        return index

    def start_task(self, key: TaskKey):
        """Triggered when a task is started."""
        self._assert_initialized()
        if not self._is_tracked_task(key):
            return

        # Mark the task at the (x, y) slice as started
        chunk_index = tuple(key[-2:])
        self._chunks[chunk_index].start()

    def finish_task(self, key: TaskKey):
        """Triggered when a task is finished."""
        self._assert_initialized()
        if not self._is_tracked_task(key):
            return

        # Mark the task at the (x, y) slice as completed
        chunk_index = tuple(key[-2:])
        self._chunks[chunk_index].finish(self._current_idx)
        self._current_idx += 1
