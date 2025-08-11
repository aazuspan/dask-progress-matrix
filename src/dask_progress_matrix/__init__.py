from importlib.metadata import version

from dask_progress_matrix.progress import ProgressMatrix

__all__ = ["ProgressMatrix"]
__version__ = version(__name__)
