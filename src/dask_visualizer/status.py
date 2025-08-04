import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import dask.array
import numpy as np
from dask.layers import ArraySliceDep
from dask_visualizer.types import Graph, TaskKey


class ComputationState(Enum):
    WAITING = 0.0
    STARTED = 0.5
    COMPLETE = 1.0


@dataclass
class ComputationChunk:
    """A 2D chunk of computation."""

    tasks_remaining: int
    state: ComputationState = ComputationState.WAITING
    completed_idx: int | None = None

    def start(self):
        """Start one task of the computation."""
        self.state = ComputationState.STARTED
        self.start_time = time.time()

    def finish(self, idx: int):
        """Finish one task of the computation."""
        self.tasks_remaining -= 1
        if self.tasks_remaining == 0:
            self.completed_idx = idx
            self.state = ComputationState.COMPLETE
            self.end_time = time.time()


class ComputationStatus:
    """
    Track the computation state of a Dask array in a Numpy array.
    """

    def __init__(
        self, obj: dask.array.Array, mode: Literal["index", "elapsed"] = "index"
    ):
        self._obj = obj
        self._mode = mode

        # Track the sequential index of the last completed chunk
        self._current_idx = 0

        # An indexer from (x, y) chunk indexes to array indexes
        self._chunk_indexer = ArraySliceDep(obj.chunks[-2:])

        # A mapping from (x, y) chunk indices to computation chunks
        self._chunks: dict[tuple[int, int], ComputationChunk] = {}

        # The current integer-encoded computation state of each block
        self.state = np.zeros(self._chunk_indexer.numblocks)

        # The completed state of each block, depending on the mode
        self.completed_state = np.zeros(self._chunk_indexer.numblocks)

    def _is_tracked_task(self, key: TaskKey) -> bool:
        """
        Check whether the given task should be tracked.

        This filters out intermediate tasks.
        """
        return isinstance(key, tuple) and key[0] == self._obj.name

    def initialize(self, dsk: Graph):
        """Triggered by the start of a computation."""
        # TODO: If the computation doesn't match the passed array, there will be no task
        # chunks. Handle that.
        chunk_indexes = [tuple(k[-2:]) for k in dsk if self._is_tracked_task(k)]
        for chunk_index, num_tasks in Counter(chunk_indexes).items():
            self._chunks[chunk_index] = ComputationChunk(num_tasks)

    def start_task(self, key: TaskKey):
        """Triggered when a task is started."""
        if not self._is_tracked_task(key):
            return

        # Mark the task at the (x, y) slice as started
        chunk_index = tuple(key[-2:])
        computation = self._chunks[chunk_index]
        computation.start()

        # Mark the block's current state
        self.state[chunk_index] = computation.state.value

    def finish_task(self, key: TaskKey):
        """Triggered when a task is finished."""
        if not self._is_tracked_task(key):
            return

        # Mark the task at the (x, y) slice as completed
        chunk_index = tuple(key[-2:])
        computation = self._chunks[chunk_index]
        computation.finish(self._current_idx)

        # Update the block's current state
        self.state[chunk_index] = computation.state.value

        # Store the appropriate value in the completed state, depending on the selected
        # mode.
        if computation.state is ComputationState.COMPLETE:
            if self._mode == "index":
                self.completed_state[chunk_index] = self._current_idx
                self._current_idx += 1
            elif self._mode == "elapsed":
                self.completed_state[chunk_index] = (
                    computation.end_time - computation.start_time
                )
