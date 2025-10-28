import inspect
import pickle
from collections.abc import Callable

import pytest
from _pytest.fixtures import FixtureRequest
from distributed import Client, LocalCluster
from pytest_textual_snapshot import (
    PseudoApp,
    PseudoConsole,
    SVGImageExtension,
    node_to_report_path,
)
from rich.console import Console
from syrupy import SnapshotAssertion
from typing_extensions import Self

from dask_progress_matrix import ProgressMatrix
from dask_progress_matrix.distributed import ProgressMatrix as DistributedProgressMatrix


class CapturedProgressMatrix:
    """
    A context manager for capturing the console output of a ProgressMatrix.
    """

    def __init__(self, id: str = "test", **matrix_kwargs):
        self._progress_matrix = ProgressMatrix(**matrix_kwargs)
        self._id = id
        self._console = Console(
            record=True, force_interactive=True, force_terminal=True
        )

    def __enter__(self) -> Self:
        self._progress_matrix.__enter__()
        self._progress_matrix._display._live.console = self._console
        return self

    def __exit__(self, *args):
        self._progress_matrix.__exit__(*args)

    @property
    def svg(self) -> str:
        """The captured SVG output of the progress matrix."""
        return self._console.export_svg(unique_id=self._id, clear=False)


class CapturedDistributedProgressMatrix:
    """
    A context manager for capturing the console output of a distributed ProgressMatrix.
    """

    def __init__(self, client: Client, id: str = "test", **matrix_kwargs):
        self._progress_matrix = DistributedProgressMatrix(client, **matrix_kwargs)
        self._id = id
        self._console = Console(
            record=True, force_interactive=True, force_terminal=True
        )

    def __enter__(self) -> Self:
        self._progress_matrix.__enter__()
        self._progress_matrix._display._live.console = self._console
        return self

    def __exit__(self, *args):
        self._progress_matrix.__exit__(*args)

    @property
    def svg(self) -> str:
        """The captured SVG output of the progress matrix."""
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


@pytest.fixture
def svg_snapshot(
    snapshot: SnapshotAssertion, request: FixtureRequest
) -> Callable[[str], bool]:
    """
    A test fixture for comparing an SVG output to a previous snapshot.

    This is adapted from pytest-textual-snapshot to take advantage of their pytest hooks
    that generate nice visual SVG diffs, while allowing testing against a simple SVG
    string instead of a Textual app.

    https://github.com/Textualize/pytest-textual-snapshot
    """
    snapshot = snapshot.use_extension(SVGImageExtension)

    def compare(actual_screenshot: str) -> bool:
        """
        Compare an SVG screenshot to a previously saved version.
        """
        node = request.node
        result = snapshot == actual_screenshot

        execution_index = (
            snapshot._custom_index
            and snapshot._execution_name_index.get(snapshot._custom_index)
        ) or snapshot.num_executions - 1
        assertion_result = snapshot.executions.get(execution_index)

        snapshot_exists = (
            execution_index in snapshot.executions
            and assertion_result
            and assertion_result.final_data is not None
        )

        expected_svg_text = str(snapshot)
        full_path, line_number, name = node.reportinfo()

        # This must match the format expected by pytest-textual-snapshot
        data = (
            result,
            expected_svg_text,
            actual_screenshot,
            PseudoApp(PseudoConsole(legacy_windows=False, size=(80, 25))),
            full_path,
            line_number,
            name,
            inspect.getdoc(node.function) or "",
            None,
            snapshot_exists,
        )
        data_path = node_to_report_path(request.node)
        data_path.write_bytes(pickle.dumps(data))

        return result

    return compare
