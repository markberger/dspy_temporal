"""The thin coarse-mode workflow.

It does no DSPy work itself -- it just durably invokes the single program
activity with the configured timeouts and retry policy. The activity dispatch
lives in the shared :func:`_coarse_activity_call` coroutine (in ``execute.py``,
the single source of truth for the activity name + timeouts), which the
context-aware ``TemporalProgram.run`` also reuses. The activity is referenced
*by name*, so the workflow only pulls dspy in as sandbox passthrough.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from ..execute import _coarse_activity_call

from ..models import ProgramCallInput, ProgramCallOutput
from ..options import CallOptions


@workflow.defn(name="DSPyProgram")
class DSPyProgramWorkflow:
    @workflow.run
    async def run(self, call: ProgramCallInput) -> ProgramCallOutput:
        # Pass the deserialized call straight through (inputs were normalized
        # client-side); only *read* its options for the timeouts. We never
        # re-instantiate it -- see _coarse_activity_call on the sandbox
        # CallOptions re-validation pitfall.
        return await _coarse_activity_call(call, call.options or CallOptions())
