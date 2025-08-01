import matplotlib.pyplot as plt
import numpy as np
from dask_visualizer.status import ArrayComputationStatus
from matplotlib import colormaps
from numpy.typing import NDArray
from PIL import Image
from rich.live import Live
from rich_pixels import Pixels


class ArrayComputationDisplay:
    def __init__(self, array_shape, height: int = 20, cmap: str = "viridis"):
        self._height = height
        self._width = self._compute_width(array_shape)
        self._cmap = cmap
        self._live = Live()

    def _compute_width(self, array_shape):
        array_height, array_width = array_shape[-2:]
        array_aspect = array_height / array_width
        return int(self._height / array_aspect)

    def update(self, status: ArrayComputationStatus):
        img = self._generate_image(status.state)
        self._live.update(img)

    def _generate_image(self, array: NDArray) -> Pixels:
        # TODO: Set these more intelligently based on the expected min and max
        vmin = np.nanmin(array)
        vmax = np.nanmax(array)

        # This is obviously very inefficient rebuilding the whole RGB image every update
        image = Image.fromarray(
            visualize_array(array, cmap_name=self._cmap, vmin=vmin, vmax=vmax)
        )

        return Pixels.from_image(image, resize=(self._width, self._height))

    def __enter__(self):
        self._live.__enter__()

    def __exit__(self, *args):
        self._live.__exit__(*args)


def visualize_array(
    arr: np.ndarray, cmap_name: str = "viridis", vmin=None, vmax=None
) -> np.ndarray:
    """
    Convert a 2D NumPy array to an RGBA image using a matplotlib colormap.
    """
    vmin = vmin if vmin is not None else np.nanmin(arr)
    vmax = vmax if vmax is not None else np.nanmax(arr)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = colormaps.get_cmap(cmap_name)
    img = cmap(norm(arr))
    return (img * 255).astype(np.uint8)
