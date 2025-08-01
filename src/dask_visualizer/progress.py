from __future__ import annotations

import dask.array
from dask.diagnostics import Callback
from dask_visualizer.display import ArrayComputationDisplay
from dask_visualizer.status import ArrayComputationStatus
from dask_visualizer.types import Graph, State, TaskKey
from numpy.typing import NDArray


class ProgressMatrix(Callback):
    # https://docs.dask.org/en/stable/diagnostics-local.html#custom-callbacks
    def __init__(
        self, obj: dask.array.Array, *, cmap: str = "viridis", height: int = 20
    ):
        self._status = ArrayComputationStatus(obj)
        self._display = ArrayComputationDisplay(obj.shape, cmap=cmap, height=height)

    def _start(self, dsk: Graph):
        self._status.initialize(dsk)
        self._display.update(self._status)

    def _pretask(self, key: TaskKey, dsk: Graph, state: State):
        self._status.start_task(key)
        self._display.update(self._status)

    def _posttask(
        self, key: TaskKey, result: NDArray, dsk: Graph, state: State, id: int
    ):
        self._status.finish_task(key)
        self._display.update(self._status)

    def _finish(self, dsk: Graph, state: State, errored: bool):
        self._status.finish()
        self._display.update(self._status)

    def __enter__(self):
        super().__enter__()
        self._display.__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        self._display.__exit__(*args)
