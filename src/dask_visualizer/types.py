from typing import Any, Unpack

TaskKey = str | tuple[str, Unpack[tuple[int, ...]]]
Graph = dict[TaskKey, Any]
State = dict[str, Any]
