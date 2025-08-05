from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dask.array
from dask.diagnostics import Callback
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr

from dask_visualizer.display import ComputationDisplay
from dask_visualizer.status import ComputationStatus
from dask_visualizer.types import Graph, State, TaskKey
from dask_visualizer.utils import extract_dask_array


class ProgressMatrix(Callback):
    """
    A progress matrix for tracking computations of 2D and 3D Dask objects by chunk.
    """

    def __init__(
        self,
        obj: dask.array.Array | xr.DataArray | xr.Dataset,
        *,
        cmap: str = "viridis",
        width: int = 20,
        mode: Literal["index", "elapsed"] = "index",
    ):
        self._obj = extract_dask_array(obj)
        self._mode = mode
        # The (x, y) shape of the object in chunks
        shape = [len(chunk) for chunk in obj.chunks[-2:]]
        self._status = ComputationStatus(shape, mode=mode)
        self._display = ComputationDisplay(shape, mode=mode, cmap=cmap, width=width)

        # Tasks will be registered when a computation is started within the progress
        # context.
        self._tracked_tasks: list[TaskKey] = []

    def _start(self, dsk: Graph):
        """
        When a computation graph is received, initialize the status and display.
        """
        # Identify the tasks that should be tracked when computing the providing Dask
        # object. If a different Dask object is computed in this context, it will return
        # no tasks and we should avoid displaying an empty progress matrix.
        self._tracked_tasks = [k for k in dsk if self._is_tracked_task(k)]
        if not self._tracked_tasks:
            return

        self._status.initialize(self._tracked_tasks)
        self._display.update(self._status.state)

    def _pretask(self, key: TaskKey, dsk: Graph, state: State):
        if key not in self._tracked_tasks:
            return

        self._status.start_task(key)
        self._display.update(self._status.state)

    def _posttask(
        self, key: TaskKey, result: NDArray, dsk: Graph, state: State, id: int
    ):
        if key not in self._tracked_tasks:
            return

        self._status.finish_task(key)
        self._display.update(self._status.state)

    def _finish(self, dsk: Graph, state: State, errored: bool):
        # If we're not currently tracking any tasks, we must be computing a different
        # Dask object and shouldn't display an empty progress matrix.
        if not self._tracked_tasks:
            return

        self._display.update(
            self._status.completed_state,
            complete=True,
        )

    def __enter__(self):
        super().__enter__()
        self._display.__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self._display.__exit__(*args)

    def _is_tracked_task(self, key: TaskKey) -> bool:
        """
        Check whether the given task should be tracked.

        This filters out tasks that are intermediate or unrelated to the tracked Dask
        object. Only tasks that contribute directly to the final Dask object will return
        True.
        """
        return isinstance(key, tuple) and key[0] == self._obj.name
