from __future__ import annotations

from typing import Literal

from dask.diagnostics import Callback
from dask_visualizer.display import ComputationDisplay
from dask_visualizer.status import ComputationStatus
from dask_visualizer.types import Graph, State, TaskKey
from dask_visualizer.utils import get_chunk_shape, get_terminal_tasks
from numpy.typing import NDArray


class ProgressMatrix(Callback):
    """
    A progress matrix for tracking computations of 2D and 3D Dask objects by chunk.
    """

    def __init__(
        self,
        *,
        cmap: str = "viridis",
        width: int = 20,
        mode: Literal["index", "elapsed"] = "index",
        show_legend: bool = True,
    ):
        self._mode = mode
        self._cmap = cmap
        self._width = width
        self._show_legend = show_legend

        # Tasks will be registered when a computation is started within the progress
        # context.
        self._terminal_tasks: list[TaskKey] = []

    def _start(self, dsk: Graph):
        """
        When a computation graph is received, initialize the status and display.
        """
        # Register the terminal tasks that will correspond to chunks in the output
        # array for this computation.
        self._terminal_tasks = get_terminal_tasks(dsk)
        shape = get_chunk_shape(self._terminal_tasks)

        self._status = ComputationStatus(
            self._terminal_tasks, shape=shape, mode=self._mode
        )

        self._display = ComputationDisplay(
            shape=shape,
            mode=self._mode,
            cmap=self._cmap,
            width=self._width,
            show_legend=self._show_legend,
        )

        self._display.__enter__()
        self._display.update(self._status.state)

    def _pretask(self, key: TaskKey, dsk: Graph, state: State):
        if key not in self._terminal_tasks:
            return

        self._status.start_task(key)
        self._display.update(self._status.state)

    def _posttask(
        self, key: TaskKey, result: NDArray, dsk: Graph, state: State, id: int
    ):
        if key not in self._terminal_tasks:
            return

        self._status.finish_task(key)
        self._display.update(self._status.state)

    def _finish(self, dsk: Graph, state: State, errored: bool):
        self._display.update(
            self._status.completed_state,
            complete=True,
        )

    def __exit__(self, *args):
        super().__exit__(*args)
        self._display.__exit__(*args)
