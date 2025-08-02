import dask
import matplotlib.pyplot as plt
import numpy as np
from dask_visualizer.status import ComputationStatus
from matplotlib import colormaps
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from PIL import Image
from rich.console import Group
from rich.live import Live
from rich.text import Text
from rich_pixels import Pixels


class ComputationDisplay:
    def __init__(self, obj: dask.array.Array, height: int = 20, cmap: str = "viridis"):
        self._obj = obj
        self._height = height
        self._width = self._compute_width(obj.shape)
        self._cmap = colormaps.get_cmap(cmap)
        self._legend = self._generate_legend()
        self._live = Live()

    def _compute_width(self, array_shape):
        array_height, array_width = array_shape[-2:]
        array_aspect = array_height / array_width
        return int(self._height / array_aspect)

    def update(self, status: ComputationStatus):
        img = self._generate_image(status.state, vmin=0, vmax=2)
        panel = Group(Text(self._obj.name) + "\n" + self._legend + "\n", img)
        self._live.update(panel)

    def _generate_image(self, array: NDArray, vmin: float, vmax: float) -> Pixels:
        # This is obviously very inefficient rebuilding the whole RGB image every update
        image = Image.fromarray(
            visualize_array(array, cmap=self._cmap, vmin=vmin, vmax=vmax)
        )

        return Pixels.from_image(image, resize=(self._width, self._height))

    def __enter__(self):
        self._live.__enter__()

    def __exit__(self, *args):
        self._live.__exit__(*args)

    def _generate_legend(self) -> Text:
        """
        Generate a legend for the colormap.
        """

        def _get_color(pixel) -> str | None:
            r, g, b, a = [int(p * 255) for p in pixel]
            return f"rgb({r},{g},{b})"

        colors = {
            "Waiting": _get_color(self._cmap(0.0)),
            "Started": _get_color(self._cmap(0.5)),
            "Finished": _get_color(self._cmap(1.0)),
        }

        labels = [
            Text("  ", style=f"on {color}") + Text(f" {state} ", style="on default")
            for state, color in colors.items()
        ]
        return Text(" ").join(labels)


def visualize_array(
    arr: np.ndarray, cmap: Colormap, vmin=None, vmax=None
) -> np.ndarray:
    """
    Convert a 2D NumPy array to an RGBA image using a matplotlib colormap.
    """
    vmin = vmin if vmin is not None else np.nanmin(arr)
    vmax = vmax if vmax is not None else np.nanmax(arr)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    img = cmap(norm(arr))
    return (img * 255).astype(np.uint8)
