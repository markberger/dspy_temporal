"""``WorkflowTool`` -- routes each ReAct tool call to a Temporal activity.

ReAct's async loop awaits ``self.tools[name].acall(**args)``. Wrapping each tool
in a ``dspy.Tool`` subclass whose ``acall`` dispatches to ``dspy_tool_call`` makes
every tool execution its own durable, independently-retried activity -- while the
loop control (which tool, with what args, when to finish) stays deterministic in
the workflow.

The wrapper copies the original's metadata (``name``/``desc``/``args``/...), so
ReAct's already-rendered instructions stay valid and anything that introspects
``program.tools[name]`` still sees a faithful tool. Only the *execution* moves.

Loaded into the workflow via ``imports_passed_through`` (host code); keep it
replay-safe -- ``workflow.execute_activity`` and pure data shaping only.
"""

from __future__ import annotations

from typing import Any

import dspy
from temporalio import workflow

from ..models import ToolCallInput, ToolCallOutput
from ..options import CallOptions
from ..serde import json_safe

TOOL_ACTIVITY_NAME = "dspy_tool_call"


class WorkflowTool(dspy.Tool):
    """A ``dspy.Tool`` whose call runs in a ``dspy_tool_call`` activity."""

    def __init__(self, original: dspy.Tool, *, program: str, options: CallOptions | None = None):
        super().__init__(
            func=original.func,
            name=original.name,
            desc=original.desc,
            args=original.args,
            arg_types=original.arg_types,
        )
        # Stored as pydantic private attributes (leading underscore); the
        # activity rebuilds the program by name and looks the tool up there.
        self._program = program
        self._options = options or CallOptions()

    async def acall(self, **kwargs: Any) -> Any:
        out = await workflow.execute_activity(
            TOOL_ACTIVITY_NAME,
            ToolCallInput(
                program=self._program,
                tool_name=self.name,
                args=json_safe(kwargs),
            ),
            result_type=ToolCallOutput,
            start_to_close_timeout=self._options.start_to_close_timeout(),
            retry_policy=self._options.retry_policy(),
        )
        return out.observation
