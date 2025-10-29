from __future__ import annotations

import asyncio
import sys
import weakref
from contextlib import suppress
from timeit import default_timer
from typing import TYPE_CHECKING, Any, Literal, TextIO

from distributed.client import futures_of
from distributed.comm import connect
from distributed.comm.core import CommClosedError
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.protocol.pickle import dumps
from distributed.utils import LoopRunner
import dask.utils

from dask_progress_matrix.display import ComputationDisplay
from dask_progress_matrix.status import ComputationStatus
from dask_progress_matrix.utils import get_chunk_shape, index_2d_from_key

if TYPE_CHECKING:
    from distributed import Client, Scheduler


class _ProgressTrackerPlugin(SchedulerPlugin):
    """
    Scheduler plugin that tracks progress of terminal tasks.
    
    This runs on the scheduler and updates state based on task transitions.
    Follows the design pattern of distributed.diagnostics.progress.Progress.
    """

    def __init__(self, keys, scheduler):
        self.scheduler = scheduler
        self.keys = {k.key if hasattr(k, "key") else k for k in keys}
        self.all_keys = self.keys.copy()
        
        self.terminal_tasks: set[tuple] = set()
        self.task_states: dict[tuple, str] = {}
        self.initialized = False
        self.status = "running"

    async def setup(self):
        """Initialize after scheduler has tasks"""
        keys = self.keys

        # Wait for keys to be in scheduler
        while not keys.issubset(self.scheduler.tasks):
            await asyncio.sleep(0.05)

        # Get all dependent tasks
        tasks = [self.scheduler.tasks[k] for k in keys]
        
        # Extract terminal tasks from the computation graph
        self.scheduler.add_plugin(self)
        self.initialized = True

    def update_graph(
        self,
        scheduler: Scheduler,
        *,
        client: str,
        keys: set,
        tasks: list,
        **kwargs,
    ):
        """Called when new tasks are added to the scheduler"""
        # Extract terminal tasks (those with tuple keys that represent chunks)
        terminal_tasks = [
            task for task in tasks if isinstance(task, tuple) and len(task) >= 2
        ]

        if terminal_tasks:
            for task in terminal_tasks:
                if task not in self.terminal_tasks:
                    self.terminal_tasks.add(task)
                    self.task_states[task] = "waiting"

    def transition(self, key: Any, start: str, finish: str, *args, **kwargs):
        """Called when a task transitions between states"""
        if key not in self.terminal_tasks:
            return

        # Track state changes
        if start in ("released", "waiting") and finish == "processing":
            self.task_states[key] = "processing"
        elif finish == "memory":
            self.task_states[key] = "complete"
            
            # Check if all complete
            if all(state == "complete" for state in self.task_states.values()):
                self.status = "finished"

    def get_state(self):
        """Return current state for client to query"""
        return {
            "terminal_tasks": list(self.terminal_tasks),
            "task_states": dict(self.task_states),
            "initialized": self.initialized and bool(self.terminal_tasks),
            "status": self.status,
        }


class ProgressMatrix:
    """
    A 2D progress matrix for tracking computations of Dask objects by chunk on
    distributed schedulers.

    This class follows the design pattern of Dask's built-in distributed progress bar,
    using a feed-based communication mechanism for real-time updates.

    Parameters
    ----------
    futures : list of futures
        The futures to track. Can be a single future or list of futures.
    interval : str, default "100ms"
        Interval for receiving updates from scheduler.
    cmap : str, default "viridis"
        The colormap to use for the progress matrix display.
    scale : int, optional
        The height of each chunk in the progress matrix, in terminal characters.
    mode : {"index", "elapsed"}, default "index"
        The type of summary displayed after a finished computation.
    show_legend : bool, default True
        If true, a legend will be displayed on top of the progress matrix.
    target_width : int, default 24
        The desired width in characters to render the matrix.
    out : TextIO, optional
        File object for output. Defaults to sys.stdout.

    Examples
    --------
    Track progress of a computation:

    >>> from distributed import Client
    >>> from dask_progress_matrix.distributed import ProgressMatrix
    >>> import dask.array as da
    >>> client = Client()  # doctest: +SKIP
    >>> x = da.random.random((128, 128), chunks=(8, 8))  # doctest: +SKIP
    >>> future = client.compute(x)  # doctest: +SKIP
    >>> ProgressMatrix(future, cmap="inferno", scale=1)  # doctest: +SKIP
    """

    __loop = None

    def __init__(
        self,
        futures,
        interval: str = "100ms",
        *,
        cmap: str = "viridis",
        mode: Literal["index", "elapsed"] = "index",
        scale: int | None = None,
        target_width: int = 24,
        show_legend: bool = True,
        out: TextIO | None = None,
    ):
        # Get futures
        self.futures = futures_of(futures)
        if not isinstance(self.futures, (set, list)):
            self.futures = [self.futures]
        
        # Get client and scheduler info
        self.client = None
        self.scheduler = None
        for key in self.futures:
            if hasattr(key, "client"):
                self.client = weakref.ref(key.client)
                self.scheduler = key.client.scheduler.address
                break

        if not self.client:
            raise ValueError("Could not determine client from futures")

        self.interval = dask.utils.parse_timedelta(interval, default="s")

        self._cmap = cmap
        self._mode = mode
        self._scale = scale
        self._target_width = target_width
        self._show_legend = show_legend
        self._out = out or sys.stdout

        self._display: ComputationDisplay | None = None
        self._status: ComputationStatus | None = None
        self._terminal_tasks: set[tuple] = set()
        self._computation_active = False
        self._start_time = default_timer()
        self._last_response = None

        # Start listening
        self._loop_runner = LoopRunner()
        self._loop_runner.run_sync(self.listen)

    @property
    def elapsed(self):
        return default_timer() - self._start_time

    async def listen(self):
        """Main listening loop - receives updates from scheduler via feed"""
        
        # Extract just the keys from futures to avoid pickling issues
        keys = [f.key if hasattr(f, 'key') else f for f in self.futures]
        
        async def setup(scheduler):
            """Setup function that runs on scheduler"""
            # Recreate futures from keys on scheduler side
            p = _ProgressTrackerPlugin(keys, scheduler)
            await p.setup()
            return p

        def function(scheduler, p):
            """Function called periodically to get state from plugin"""
            return p.get_state()

        # Connect to scheduler
        comm = await connect(
            self.scheduler,
            **self.client().connection_args,
        )

        # Set up feed operation
        await comm.write(
            {
                "op": "feed",
                "setup": dumps(setup),
                "function": dumps(function),
                "interval": self.interval,
            },
            serializers=self.client()._serializers,
        )

        # Listen for updates
        while True:
            try:
                response = await comm.read(
                    deserializers=self.client()._deserializers
                )
            except CommClosedError:
                break

            self._last_response = response
            
            # Process the update
            self._process_update(response)
            
            # Check if finished
            if response.get("status") == "finished":
                await comm.close()
                self._draw_stop()
                break

    def _process_update(self, state: dict):
        """Process state update from scheduler"""
        if not state or not state.get("initialized"):
            return

        terminal_tasks = state.get("terminal_tasks", [])
        task_states = state.get("task_states", {})

        # Convert task keys from lists to tuples if needed
        terminal_tasks = [
            tuple(t) if isinstance(t, list) else t for t in terminal_tasks
        ]
        task_states = {
            tuple(k) if isinstance(k, list) else k: v
            for k, v in task_states.items()
        }

        # Initialize display if this is the first update with terminal tasks
        if not self._computation_active and terminal_tasks:
            self._initialize_display(terminal_tasks)

        if not self._computation_active:
            return

        # Update all task states
        for task, state_name in task_states.items():
            if task not in self._terminal_tasks:
                continue

            task_index = index_2d_from_key(task)
            
            # Apply state transitions
            if state_name == "processing":
                self._status.start_task(task_index)
            elif state_name == "complete":
                self._status.finish_task(task_index)

        # Update display
        if self._display:
            self._display.update(self._status.state)

    def _initialize_display(self, terminal_tasks: list):
        """Initialize the progress matrix display"""
        if not terminal_tasks:
            return

        self._terminal_tasks = set(terminal_tasks)
        task_indexes = [index_2d_from_key(k) for k in terminal_tasks]
        shape = get_chunk_shape(task_indexes)

        self._status = ComputationStatus(task_indexes, shape=shape, mode=self._mode)

        self._display = ComputationDisplay(
            cmap=self._cmap,
            mode=self._mode,
            scale=self._scale,
            target_width=self._target_width,
            show_legend=self._show_legend,
            out=self._out,
        )
        self._display.initialize(shape)
        self._display.__enter__()
        self._display.update(self._status.state)
        self._computation_active = True

    def _draw_stop(self):
        """Called when computation finishes"""
        if self._display and self._computation_active:
            self._display.update(
                self._status.completed_state,
                complete=True,
            )
            self._display.__exit__()
            self._computation_active = False

    def __del__(self):
        """Cleanup on deletion"""
        with suppress(AttributeError):
            if hasattr(self, "_loop_runner"):
                self._loop_runner.stop()
