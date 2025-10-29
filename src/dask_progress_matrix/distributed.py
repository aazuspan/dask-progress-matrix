from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, TextIO

from distributed.diagnostics.plugin import SchedulerPlugin

from dask_progress_matrix.display import ComputationDisplay
from dask_progress_matrix.status import ComputationStatus
from dask_progress_matrix.utils import get_chunk_shape, index_2d_from_key

if TYPE_CHECKING:
    from distributed import Client, Scheduler


class _ProgressPlugin(SchedulerPlugin):
    """
    Scheduler plugin that tracks task state.

    This plugin runs on the scheduler and must be serializable.
    It stores state that can be queried by the client.
    """

    def __init__(self):
        self._terminal_tasks: set[tuple] = set()
        self._started_tasks: set[tuple] = set()
        self._finished_tasks: set[tuple] = set()
        self._computation_initialized = False

    def update_graph(
        self,
        scheduler: Scheduler,
        *,
        client: str,
        keys: set,
        tasks: list,
        **kwargs,
    ):
        """Called when new tasks are added to the scheduler."""
        # Extract terminal tasks (those with tuple keys that represent chunks)
        terminal_tasks = [
            task for task in tasks if isinstance(task, tuple) and len(task) >= 2
        ]

        if not terminal_tasks:
            return

        # Initialize tracking for this computation if it's a new one
        if not self._computation_initialized or not self._terminal_tasks:
            self._terminal_tasks = set(terminal_tasks)
            self._started_tasks = set()
            self._finished_tasks = set()
            self._computation_initialized = True

    def transition(self, key: Any, start: str, finish: str, *args, **kwargs):
        """Called when a task transitions between states."""
        if key not in self._terminal_tasks:
            return

        # Track when a task starts processing
        if start in ("released", "waiting") and finish == "processing":
            self._started_tasks.add(key)

        # Track when a task completes
        elif finish == "memory":
            self._finished_tasks.add(key)


class ProgressMatrix:
    """
    A 2D progress matrix for tracking computations of Dask objects by chunk on
    distributed schedulers.

    This class is designed to work with `distributed.Client` and tracks task
    progress via the scheduler plugin mechanism.

    Parameters
    ----------
    client : distributed.Client
        The Dask distributed client to attach this progress matrix to.
    cmap : str, default "viridis"
        The colormap to use for the progress matrix display.
    scale : int, optional
        The height of each chunk in the progress matrix, in terminal characters. If not
        provided, scale will be calculated to render nearest to the provided
        `target_width`.
    mode : {"index", "elapsed"}, default "index"
        The type of summary displayed after a finished computation:
        - `"index"`: Shows the order that each chunk was computed in.
        - `"elapsed"`: Shows the elapsed time between starting and ending each chunk.
    show_legend : bool, default True
        If true, a legend will be displayed on top of the progress matrix to explain
        color encodings.
    target_width : int, default 24
        The desired width in characters to render the matrix. If `scale` is not
        provided, it will be calculated to render blocks as close as possible to the
        target width. Because each block is rendered to a mimimum of 2 characters, it
        may not be possible to render to the exact target width. Ignored if `scale` is
        provided.
    out : TextIO, optional
        File object into which the the progress matrix will be written. If not provided,
        `sys.stdout` is used.

    Examples
    --------

    Use as a context manager with a distributed client:

    >>> from distributed import Client
    >>> from dask_progress_matrix.distributed import ProgressMatrix
    >>> import dask.array as da
    >>> client = Client()  # doctest: +SKIP
    >>> # doctest: +SKIP
    >>> with ProgressMatrix(client, cmap="inferno", scale=1, mode="index"):
    ...     x = da.random.random((128, 128), chunks=(8, 8))
    ...     x.compute()
    """

    def __init__(
        self,
        client: Client,
        *,
        cmap: str = "viridis",
        mode: Literal["index", "elapsed"] = "index",
        scale: int | None = None,
        target_width: int = 24,
        show_legend: bool = True,
        out: TextIO | None = None,
    ):
        self._client = client
        self._cmap = cmap
        self._mode = mode
        self._scale = scale
        self._target_width = target_width
        self._show_legend = show_legend
        self._out = out

        self._display = ComputationDisplay(
            cmap=self._cmap,
            mode=self._mode,
            scale=self._scale,
            target_width=self._target_width,
            show_legend=self._show_legend,
            out=self._out,
        )

        # Track terminal tasks for the current computation
        self._terminal_tasks: set[tuple] = set()
        self._status: ComputationStatus | None = None
        self._computation_active = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()
        
        # Track which tasks we've already processed to avoid duplicate updates
        self._processed_started_tasks: set[tuple] = set()
        self._processed_finished_tasks: set[tuple] = set()

    def __enter__(self):
        """Register the plugin when entering context."""
        # Create and register the plugin
        plugin = _ProgressPlugin()
        self._client.register_plugin(plugin, name="progress-matrix")

        # Start monitoring thread
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        return self

    def __exit__(self, *args):
        """Unregister the plugin and clean up display when exiting context."""
        try:
            # Stop monitoring thread
            self._stop_monitoring.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)

            # Do one final poll to catch any completed tasks that finished
            # after the last poll but before the thread stopped
            if self._computation_active:
                with contextlib.suppress(Exception):
                    self._poll_plugin_state()

            # Clean up display if still active
            if self._computation_active and self._display:
                self._display.__exit__(*args)
                self._computation_active = False
        finally:
            self._client.unregister_scheduler_plugin(name="progress-matrix")

    def _monitor_loop(self):
        """Monitor the plugin state and update the display."""
        while not self._stop_monitoring.is_set():
            # Ignore errors to prevent crashes
            with contextlib.suppress(Exception):
                self._poll_plugin_state()
            time.sleep(0.1)  # Poll every 100ms

    def _poll_plugin_state(self):
        """Poll the plugin state from the scheduler."""

        def get_plugin_state(dask_scheduler):
            plugin = dask_scheduler.plugins.get("progress-matrix")
            if plugin:
                return {
                    "terminal_tasks": list(plugin._terminal_tasks),
                    "started_tasks": list(plugin._started_tasks),
                    "finished_tasks": list(plugin._finished_tasks),
                    "initialized": plugin._computation_initialized,
                }
            return None

        state = self._client.run_on_scheduler(get_plugin_state)
        if not state or not state.get("initialized"):
            return

        # Initialize if we haven't yet
        if not self._computation_active and state["terminal_tasks"]:
            self._initialize_display(state["terminal_tasks"])

        if not self._computation_active:
            return

        # Update task states
        started_tasks = set(
            tuple(t) if isinstance(t, list) else t for t in state["started_tasks"]
        )
        finished_tasks = set(
            tuple(t) if isinstance(t, list) else t for t in state["finished_tasks"]
        )

        # Track which tasks have changed state (only process new state changes)
        new_started_tasks = started_tasks - self._processed_started_tasks
        new_finished_tasks = finished_tasks - self._processed_finished_tasks
        
        for task in new_started_tasks:
            if task in self._terminal_tasks:
                self._status.start_task(index_2d_from_key(task))
                self._processed_started_tasks.add(task)

        for task in new_finished_tasks:
            if task in self._terminal_tasks:
                self._status.finish_task(index_2d_from_key(task))
                self._processed_finished_tasks.add(task)

        # Update display
        self._display.update(self._status.state)

        # Check if computation is complete
        if len(finished_tasks) == len(self._terminal_tasks):
            self._display.update(
                self._status.completed_state,
                complete=True,
            )
            self._display.__exit__()
            self._computation_active = False

    def _initialize_display(self, terminal_tasks: list):
        """Initialize the progress matrix for a new computation."""
        if not terminal_tasks:
            return

        # Convert from list to tuple if needed
        terminal_tasks = [
            tuple(t) if isinstance(t, list) else t for t in terminal_tasks
        ]

        self._terminal_tasks = set(terminal_tasks)
        task_indexes = [index_2d_from_key(k) for k in terminal_tasks]
        shape = get_chunk_shape(task_indexes)

        self._status = ComputationStatus(task_indexes, shape=shape, mode=self._mode)

        # Reset processed task tracking for new computation
        self._processed_started_tasks = set()
        self._processed_finished_tasks = set()

        self._display.initialize(shape)
        self._display.__enter__()
        self._display.update(self._status.state)
        self._computation_active = True
