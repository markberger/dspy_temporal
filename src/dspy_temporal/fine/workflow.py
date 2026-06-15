"""The fine-mode workflow: orchestrate the program, delegate every call.

Unlike the coarse workflow (a thin one-activity invoker), this one *runs the
program* under a ``dspy.context`` that swaps in a ``WorkflowLM`` and
activity-backed tools, so the orchestration (adapter format/parse, the ReAct
loop) executes as deterministic Python in the workflow while each LM call and
tool call becomes its own activity. Completed calls are recorded in Temporal
history, so a crash + replay resumes from the last finished step.

The orchestration body itself lives in :func:`dspy_temporal.execute.execute_fine`
(shared with the context-aware ``TemporalProgram.run`` so a user can compose a
program into their own workflow). This class is the thin generic
wrapper: it runs that body and adapts the returned ``Prediction`` to the
over-the-wire ``ProgramCallOutput`` shape. See ``execute.py`` for the full
replay-safety and import-passthrough rationale.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from ..execute import execute_fine
    from ..serde import prediction_to_dict

from ..models import ProgramCallInput, ProgramCallOutput
from ..options import CallOptions


@workflow.defn(name="DSPyProgramFine")
class DSPyProgramFineWorkflow:
    @workflow.run
    async def run(self, call: ProgramCallInput) -> ProgramCallOutput:
        # The orchestration body lives in execute_fine (shared with the
        # context-aware TemporalProgram.run). This thin wrapper only adapts the
        # returned Prediction to the over-the-wire ProgramCallOutput shape.
        prediction = await execute_fine(
            call.program, call.inputs, call.options or CallOptions()
        )
        return ProgramCallOutput(
            prediction=prediction_to_dict(prediction),
            lm_usage=prediction.get_lm_usage() or None,
        )
