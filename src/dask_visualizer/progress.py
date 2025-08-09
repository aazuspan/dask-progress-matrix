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

    Parameters
    ----------
    cmap : str, default "viridis"
        The colormap to use for the progress matrix display.
    scale : int, optional
        The height of each chunk in the progress matrix, in terminal characters. If not
        provided, scale will be calculated to render nearest to the provided
        `target_width`.
    mode : {"index", "elapsed"}, default "index"
        The type of summary displayed after a finished computation:
        - `"index"`: Shows the order that each chunk was computed in.
        - `"elapsed"`: Shows the elapsed time between starting and ending each chunk.
    show_legend : bool, default True
        If true, a legend will be displayed on top of the progress matrix to explain
        color encodings.
    target_width : int, default 24
        The desired width in characters to render the matrix. If `scale` is not
        provided, it will be calculated to render blocks as close as possible to the
        target width. Because each block is rendered to a mimimum of 2 characters, it
        may not be possible to render to the exact target width. Ignored if `scale` is
        provided.

    Examples
    --------

    Run a computation within a `ProgressMatrix` context to visualize the progress of
    each chunk.

    >>> import dask.array as da
    >>> from dask_visualizer import ProgressMatrix
    >>> with ProgressMatrix(cmap="inferno", scale=1, mode="index"):
    ...     x = da.random.random((128, 128), chunks=(8, 8))
    ...     x.compute()
    """

    def __init__(
        self,
        *,
        cmap: str = "viridis",
        mode: Literal["index", "elapsed"] = "index",
        scale: int | None = None,
        target_width: int = 24,
        show_legend: bool = True,
    ):
        self._cmap = cmap
        self._mode = mode
        self._scale = scale
        self._target_width = target_width
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
            cmap=self._cmap,
            mode=self._mode,
            scale=self._scale,
            target_width=self._target_width,
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
