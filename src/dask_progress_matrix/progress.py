from __future__ import annotations

from typing import Literal, TextIO, TypeGuard

from dask.diagnostics import Callback
from numpy.typing import NDArray

from dask_progress_matrix.display import ComputationDisplay
from dask_progress_matrix.status import ComputationStatus
from dask_progress_matrix.types import Graph, IndexedTaskKey, State, TaskKey
from dask_progress_matrix.utils import (
    get_chunk_shape,
    get_terminal_tasks,
    index_2d_from_key,
)


class ProgressMatrix(Callback):
    """
    A 2D progress matrix for tracking computations of Dask objects by chunk.

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
    out : TextIO, optional
        File object into which the the progress matrix will be written. If not provided,
        `sys.stdout` is used.

    Examples
    --------

    Run a computation within a `ProgressMatrix` context to visualize the progress of
    each chunk.

    >>> import dask.array as da
    >>> from dask_progress_matrix import ProgressMatrix
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
        out: TextIO | None = None,
    ):
        self._cmap = cmap
        self._mode = mode
        self._scale = scale
        self._target_width = target_width
        self._show_legend = show_legend
        self._out = out

        self._display = ComputationDisplay(
            cmap=self._cmap,
            mode=self._mode,
            scale=self._scale,
            target_width=self._target_width,
            show_legend=self._show_legend,
            out=self._out,
        )

        # Tasks will be registered when a computation is started within the progress
        # context.
        self._terminal_tasks: set[IndexedTaskKey] = set()

    def _start(self, dsk: Graph):
        """
        When a computation graph is received, initialize the status and display.
        """
        # Register the terminal tasks that will correspond to chunks in the output
        # array for this computation.
        self._terminal_tasks = get_terminal_tasks(dsk)

        # It's possible for a computation to have no indexed tasks, e.g. a single chunk
        # Dask array. In that case, we shouldn't display anything.
        if not self._terminal_tasks:
            return

        task_indexes = [index_2d_from_key(k) for k in self._terminal_tasks]
        shape = get_chunk_shape(task_indexes)

        self._status = ComputationStatus(task_indexes, shape=shape, mode=self._mode)

        self._display.initialize(shape)
        self._display.__enter__()
        self._display.update(self._status.state)

    def _pretask(self, key: TaskKey, dsk: Graph, state: State):
        if not self._is_terminal_task(key):
            return

        changed_state = self._status.start_task(index_2d_from_key(key))
        if changed_state:
            self._display.update(self._status.state)

    def _posttask(
        self, key: TaskKey, result: NDArray, dsk: Graph, state: State, id: int
    ):
        if not self._is_terminal_task(key):
            return

        changed_state = self._status.finish_task(index_2d_from_key(key))
        if changed_state:
            self._display.update(self._status.state)

    def _finish(self, dsk: Graph, state: State, errored: bool):
        # If there were no terminal tasks, the matrix was never initialized.
        if not self._terminal_tasks:
            return

        self._display.update(
            self._status.completed_state,
            complete=True,
        )
        # Exit the display after finishing the computation to allow multiple
        # computations within a single context without interfering.
        self._display.__exit__()

    def __exit__(self, *args):
        super().__exit__(*args)
        # Exit the display context when the progress context is closed. This will have
        # no effect if the computation finished since the display already exited, but
        # gracefully stops the display if there's an error. If there was no computation,
        # the display won't be initialized.
        if getattr(self, "_display", None):
            self._display.__exit__(*args)

    def _is_terminal_task(self, key: TaskKey) -> TypeGuard[IndexedTaskKey]:
        return key in self._terminal_tasks
