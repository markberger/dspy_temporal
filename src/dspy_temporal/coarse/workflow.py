"""The thin coarse-mode workflow.

It does no DSPy work itself -- it just durably invokes the single program
activity with the configured timeouts and retry policy. The activity is
referenced *by name* (not by importing its function) so the heavy ``dspy``
import never enters the workflow sandbox.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from ..models import ProgramCallInput, ProgramCallOutput
    from ..options import CallOptions

ACTIVITY_NAME = "dspy_run_program"


@workflow.defn(name="DSPyProgram")
class DSPyProgramWorkflow:
    @workflow.run
    async def run(self, call: ProgramCallInput) -> ProgramCallOutput:
        options = call.options or CallOptions()
        return await workflow.execute_activity(
            ACTIVITY_NAME,
            call,
            result_type=ProgramCallOutput,
            start_to_close_timeout=options.start_to_close_timeout(),
            heartbeat_timeout=options.heartbeat_timeout(),
            retry_policy=options.retry_policy(),
        )
