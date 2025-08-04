from __future__ import annotations

from typing import TYPE_CHECKING

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
    # https://docs.dask.org/en/stable/diagnostics-local.html#custom-callbacks
    def __init__(
        self,
        obj: dask.array.Array | xr.DataArray | xr.Dataset,
        *,
        cmap: str = "viridis",
        height: int = 20,
    ):
        obj = extract_dask_array(obj)
        self._status = ComputationStatus(obj)
        self._display = ComputationDisplay(obj, cmap=cmap, height=height)

    def _start(self, dsk: Graph):
        self._status.initialize(dsk)
        self._display.update(self._status.state)

    def _pretask(self, key: TaskKey, dsk: Graph, state: State):
        self._status.start_task(key)
        self._display.update(self._status.state)

    def _posttask(
        self, key: TaskKey, result: NDArray, dsk: Graph, state: State, id: int
    ):
        self._status.finish_task(key)
        self._display.update(self._status.state)

    def _finish(self, dsk: Graph, state: State, errored: bool):
        self._display.update(self._status.completed_state)

    def __enter__(self):
        super().__enter__()
        self._display.__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self._display.__exit__(*args)
