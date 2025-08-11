import argparse
import runpy

from dask_progress_matrix import ProgressMatrix


def main():
    parser = argparse.ArgumentParser(
        description="Run a Python script with a Dask progress matrix."
    )
    parser.add_argument("file")
    parser.add_argument(
        "--cmap", default="viridis", help="Colormap to use for the progress matrix."
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Height of each chunk in the progress matrix.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=24,
        help="Desired width in characters to render the matrix.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="File to write the progress matrix to. Defaults to stdout.",
    )
    parser.add_argument(
        "--mode",
        choices=["index", "elapsed"],
        default="index",
        help="Mode for the progress matrix.",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Whether to hide the legend in the progress matrix.",
    )

    args = parser.parse_args()

    with ProgressMatrix(
        cmap=args.cmap,
        scale=args.scale,
        target_width=args.target_width,
        out=args.out,
        mode=args.mode,
        show_legend=not args.hide_legend,
    ):
        runpy.run_path(args.file, run_name="__main__")
