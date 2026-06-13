"""Fine-grained mode: per-LM-call and per-tool-call Temporal activities.

The workflow (:class:`DSPyProgramFineWorkflow`) runs the program's orchestration
and delegates each LM call to :func:`lm_call_activity` (via :class:`WorkflowLM`)
and each tool call to :func:`tool_call_activity` (via :class:`WorkflowTool`).
Opt in per program with ``RunConfig(mode="fine")``.
"""

from __future__ import annotations

from .activities import lm_call_activity, tool_call_activity
from .lm import WorkflowLM
from .tools import WorkflowTool
from .workflow import DSPyProgramFineWorkflow

__all__ = [
    "DSPyProgramFineWorkflow",
    "WorkflowLM",
    "WorkflowTool",
    "lm_call_activity",
    "tool_call_activity",
]
