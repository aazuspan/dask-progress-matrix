import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from dask_progress_matrix.types import ChunkIndex


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
        indexes: list[ChunkIndex],
        *,
        shape: tuple[int, int],
        mode: Literal["index", "elapsed"] = "index",
    ):
        self._mode = mode

        # Track the sequential index of the last completed chunk
        self._current_idx = 0

        # A mapping from (y, x) chunk indices to computation chunks
        self._chunks = self._initialize_chunks(indexes)

        # The current integer-encoded computation state of each chunk
        self.state = np.zeros(shape)

        # The completed state of each chunk, depending on the mode
        self.completed_state = np.zeros(shape)

    def _initialize_chunks(
        self,
        indexes: list[ChunkIndex],
    ) -> dict[ChunkIndex, ComputationChunk]:
        """Triggered by the start of a computation."""
        chunks = {}

        for chunk_index, num_tasks in Counter(indexes).items():
            chunks[chunk_index] = ComputationChunk(num_tasks)

        return chunks

    def start_task(self, chunk_index: ChunkIndex) -> bool:
        """
        Start a task and return whether the associated chunk changed state.
        """
        # Mark the task at the (y, x) index as started
        computation = self._chunks[chunk_index]
        prev_state = computation.state.value
        computation.start()

        # Mark the block's current state
        self.state[chunk_index] = computation.state.value

        return prev_state != computation.state.value

    def finish_task(self, chunk_index: ChunkIndex) -> bool:
        """
        Finish a task and return whether the associated chunk changed state.
        """
        # Mark the task at the (y, x) slice as completed
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
            return True

        return False
