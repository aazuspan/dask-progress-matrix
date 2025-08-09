import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from dask_visualizer.types import IndexedTaskKey


class ComputationState(Enum):
    WAITING = 0.0
    STARTED = 0.5
    COMPLETE = 1.0


@dataclass
class ComputationChunk:
    """A 2D chunk of computation."""

    tasks_remaining: int
    state: ComputationState = ComputationState.WAITING

    def start(self):
        """Start one task of the computation."""
        self.state = ComputationState.STARTED
        self.start_time = time.time()

    def finish(self):
        """Finish one task of the computation."""
        self.tasks_remaining -= 1
        if self.tasks_remaining == 0:
            self.state = ComputationState.COMPLETE
            self.end_time = time.time()


class ComputationStatus:
    """
    Track the computation state of a Dask array in a Numpy array.
    """

    def __init__(
        self,
        tasks: set[IndexedTaskKey],
        *,
        shape: tuple[int, int],
        mode: Literal["index", "elapsed"] = "index",
    ):
        self._mode = mode

        # Track the sequential index of the last completed chunk
        self._current_idx = 0

        # A mapping from (x, y) chunk indices to computation chunks
        self._chunks = self._initialize_chunks(tasks)

        # The current integer-encoded computation state of each chunk
        self.state = np.zeros(shape)

        # The completed state of each chunk, depending on the mode
        self.completed_state = np.zeros(shape)

    def _initialize_chunks(
        self, tasks: set[IndexedTaskKey]
    ) -> dict[tuple[int, int], ComputationChunk]:
        """Triggered by the start of a computation."""
        chunks = {}

        chunk_indexes = [k[-2:] for k in tasks]
        for chunk_index, num_tasks in Counter(chunk_indexes).items():
            chunks[chunk_index] = ComputationChunk(num_tasks)

        return chunks

    def start_task(self, key: IndexedTaskKey) -> None:
        """Triggered when a task is started."""
        # Mark the task at the (x, y) slice as started
        chunk_index = key[-2:]
        computation = self._chunks[chunk_index]
        computation.start()

        # Mark the block's current state
        self.state[chunk_index] = computation.state.value

    def finish_task(self, key: IndexedTaskKey) -> None:
        """Triggered when a task is finished."""
        # Mark the task at the (x, y) slice as completed
        chunk_index = key[-2:]
        computation = self._chunks[chunk_index]
        computation.finish()

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
