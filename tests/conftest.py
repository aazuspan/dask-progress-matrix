import pytest
from distributed import Client, LocalCluster
from rich.console import Console
from typing_extensions import Self

from dask_progress_matrix import ProgressMatrix as LocalProgressMatrix
from dask_progress_matrix.distributed import ProgressMatrix as DistributedProgressMatrix


class CapturedProgressMatrix:
    """
    A context manager for capturing the console output of a ProgressMatrix.
    """

    def __init__(self, id: str = "test", **matrix_kwargs):
        self._matrix_kwargs = matrix_kwargs
        self._id = id

    def __enter__(self) -> Self:
        self._console = Console(
            record=True, force_interactive=True, force_terminal=True
        )
        self._progress_matrix = LocalProgressMatrix(
            out=self._console.file, **self._matrix_kwargs
        )
        self._progress_matrix.__enter__()
        return self

    def __exit__(self, *args):
        self._progress_matrix.__exit__(*args)

    @property
    def svg(self) -> str:
        """The captured SVG output of the progress matrix."""
        return self._console.export_svg(unique_id=self._id, clear=False)


class CapturedDistributedProgressMatrix:
    """
    Helper for capturing the console output of a distributed ProgressMatrix.
    
    Since the distributed version doesn't use a context manager, this class
    creates the futures and passes them to ProgressMatrix, then captures output.
    """

    def __init__(self, client: Client, compute_fn, id: str = "test", **matrix_kwargs):
        self._client = client
        self._compute_fn = compute_fn
        self._matrix_kwargs = matrix_kwargs
        self._id = id
        self._console = Console(
            record=True, force_interactive=True, force_terminal=True
        )

    def run(self) -> str:
        """Run the computation and capture output"""
        # Create dask object
        dask_obj = self._compute_fn()
        
        # Get futures
        futures = self._client.compute(dask_obj)
        
        # Create progress matrix with captured output
        DistributedProgressMatrix(
            futures,
            out=self._console.file,
            **self._matrix_kwargs
        )
        
        # Return SVG
        return self._console.export_svg(unique_id=self._id, clear=False)


@pytest.fixture
def distributed_client():
    """Create a local distributed client for testing."""
    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, processes=False, silence_logs=True
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()
