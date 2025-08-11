from __future__ import annotations

from types import TracebackType
from typing import Literal, TextIO

import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.segment import Segment, Segments
from rich.style import Style
from rich.table import Table
from rich.text import Text

from dask_progress_matrix.status import ComputationState


class ComputationDisplay:
    _chunk_width = 2

    def __init__(
        self,
        *,
        mode: Literal["index", "elapsed"],
        scale: int | None = None,
        cmap: str = "viridis",
        show_legend: bool = True,
        target_width: int = 24,
        out: TextIO | None = None,
    ):
        self._mode = mode
        self._show_legend = show_legend
        self._scale = scale
        self._target_width = target_width
        self._cmap = colormaps.get_cmap(cmap)
        self._live = Live(
            console=Console(file=out),
            auto_refresh=False,
        )

    def initialize(self, shape: tuple[int, int]) -> None:
        """
        Initialize the display to a given shape.

        This must be called prior to updating the display.
        """
        self._computed_scale = self._scale or self._calculate_scale(
            shape[1], self._target_width
        )
        self._width = self._computed_scale * shape[1] * self._chunk_width
        self._legend = self._generate_legend()

    def __enter__(self):
        self._live.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ):
        # Live requires all three arguments, but ignores them
        self._live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, state: NDArray, complete=False):
        legend: RenderableType
        # When complete, display the appropriate colorbar and normalize the state for
        # drawing.
        if complete:
            legend = self._generate_colorbar(state)
            with np.errstate(divide="ignore", invalid="ignore"):
                state = state / state.max()
        else:
            legend = self._legend

        content: list[RenderableType] = [self._render_array(state)]
        if self._show_legend:
            content = [legend, Text(""), *content]

        self._live.update(Group(*content), refresh=True)

    def _generate_legend(self) -> Table:
        """Generate a legend for the colormap."""
        colors = [
            self._get_color(self._cmap(i.value))
            for i in (
                ComputationState.WAITING,
                ComputationState.STARTED,
                ComputationState.COMPLETE,
            )
        ]
        colorbar = Table(
            title="Chunk state", width=self._width, padding=0, show_edge=False, box=None
        )
        colorbar.add_column("Waiting", justify="left", header_style="not bold")
        colorbar.add_column("Started", justify="center", header_style="not bold")
        colorbar.add_column("Complete", justify="right", header_style="not bold")
        colorbar.add_row(*[Text("  ", style=f"on {color}") for color in colors])

        return colorbar

    def _generate_colorbar(self, state: NDArray) -> Group:
        """
        Generate a colorbar for the un-normalized state array.

        The colorbar title and formatting will depend on the selected mode, i.e. whether
        the final state displays the index or the elapsed time of each computed chunk.
        """
        min_val = state.min()
        max_val = state.max()

        # Indexes should be listed as integers
        if self._mode == "index":
            cbar_title = "Chunk index"
            cbar_format = ".0f"
        # Elapsed time should be listed as floats with a reasonable unit
        elif self._mode == "elapsed":
            if max_val < 0.1:
                unit = "ms"
                max_val *= 1000
                min_val *= 1000
            elif max_val > 60:
                unit = "m"
                max_val /= 60
                min_val /= 60
            else:
                unit = "s"
            cbar_title = f"Elapsed ({unit})"
            cbar_format = ".2f"

        cbar_min = format(min_val, cbar_format)
        cbar_max = format(max_val, cbar_format)

        gradient = Segments(
            [
                Segment(" ", style=Style.parse(f"on {self._get_color(self._cmap(i))}"))
                for i in np.linspace(0.0, 1.0, self._width)
            ]
        )

        colorbar = Table(
            title=cbar_title,
            width=self._width,
            padding=0,
            pad_edge=False,
            show_edge=False,
            box=None,
        )
        colorbar.add_column(cbar_min, justify="left", header_style="not bold")
        colorbar.add_column(cbar_max, justify="right", header_style="not bold")

        return Group(colorbar, gradient, Text(""))

    def _render_array(self, array: NDArray) -> Segments:
        segments = []

        for line in array:
            line_segments = []

            for block in line:
                rgba = self._cmap(block)
                c = self._get_color(rgba)
                line_segments.append(
                    Segment(
                        " " * self._chunk_width * self._computed_scale,
                        style=Style.parse(f"on {c}"),
                    )
                )

            for _ in range(self._computed_scale):
                segments += line_segments
                segments.append(Segment("\n"))

        return Segments(segments)

    def _calculate_scale(self, width_chunks: int, target_width: int) -> int:
        """
        Chose a scale that renders the given chunks nearest to the target width.
        """
        over = max(int(target_width // width_chunks / self._chunk_width), 1)
        under = max(over - 1, 1)

        # If they calculate to the same scale because the minimum width is larger than
        # the target width, return either.
        if over == under:
            return under

        # Choose the scale with the lower error from the target width, or the smaller
        # scale if they're equal.
        over_error = abs(target_width - over * width_chunks * self._chunk_width)
        under_error = abs(target_width - under * width_chunks * self._chunk_width)

        if under_error <= over_error:
            return under
        return over

    @staticmethod
    def _get_color(pixel: tuple[float, float, float, float]) -> str | None:
        """Convert an RGBA tuple in range [0, 1] to a CSS RGB string."""
        r, g, b, a = [int(p * 255) for p in pixel]
        return f"rgb({r},{g},{b})"
