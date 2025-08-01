import dask.array
import numpy as np
from dask.layers import ArraySliceDep
from dask_visualizer.types import Graph, TaskKey


class ArrayComputationStatus:
    """
    Track the computation state of a Dask array in a Numpy array.
    """

    def __init__(self, obj: dask.array.Array):
        self.obj = obj
        self.slices = ArraySliceDep(obj.chunks)

    def assert_initialized(self):
        """Ensure that the computation has been initialized."""
        if self.graph is None or self.state is None:
            raise AttributeError("The computation has not been initialized.")

    def initialize(self, dsk: Graph):
        """Triggered by the start of a computation."""
        self.graph = dsk
        self.state = np.zeros(self.obj.shape)

    def start_task(self, key: TaskKey):
        """Triggered when a task is started."""
        self.assert_initialized()
        if isinstance(key, str):
            return

        try:
            idx = self.slices[key[1:]]
        except IndexError:
            print(key)

        self.state[idx] += 1

    def finish_task(self, key: TaskKey):
        """Triggered when a task is finished."""
        self.assert_initialized()
        if isinstance(key, str):
            return

        # TODO: Only mark complete when the last task for a given slice is finished.
        # TODO: Or probably mark the percentage of tasks complete for that slice.
        # self[key] = 2

    def finish(self): ...
