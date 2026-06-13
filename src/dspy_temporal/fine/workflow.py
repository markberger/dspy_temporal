"""The fine-mode workflow: orchestrate the program, delegate every call.

Unlike the coarse workflow (a thin one-activity invoker), this one *runs the
program* -- ``await program.acall(**inputs)`` -- but under a ``dspy.context`` that
swaps in a ``WorkflowLM`` and activity-backed tools. So the program's
orchestration (adapter format/parse, the ReAct loop) executes as deterministic
Python in the workflow, while each LM call and tool call becomes its own
activity. Completed calls are recorded in Temporal history, so a crash + replay
resumes from the last finished step.

Why this is replay-safe even though heavy DSPy runs here: the *only* things that
drive workflow commands (which activities run, with what args, in what order) are
the activity results, which Temporal records and replays identically. DSPy's
local nondeterminism (history timestamps/uuids) never touches a command -- and we
suppress it anyway by running with ``callbacks=[]`` (no span emission in the
workflow; spans are emitted in the activities).

Heavy/host-state imports go through ``imports_passed_through()`` so the workflow
shares the **host** modules -- most importantly the host program registry (a
sandbox reload would build an empty one) -- and so DSPy runs as unrestricted
passthrough rather than tripping the sandbox.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import dspy

    from ..registry import default_registry
    from ..serde import prediction_to_dict
    from .lm import WorkflowLM
    from .tools import WorkflowTool

from ..models import ProgramCallInput, ProgramCallOutput
from ..options import CallOptions

# ReAct's internal "finish" marker is a no-op (lambda: "Completed."); keep it
# local to the workflow instead of paying an activity round-trip for it.
_LOCAL_TOOLS = frozenset({"finish"})


@workflow.defn(name="DSPyProgramFine")
class DSPyProgramFineWorkflow:
    @workflow.run
    async def run(self, call: ProgramCallInput) -> ProgramCallOutput:
        options = call.options or CallOptions()
        program = default_registry().build(call.program)

        # Wrap each tool (ReAct and any module exposing a `.tools` dict) so its
        # execution dispatches to the dspy_tool_call activity. The wrapper keeps
        # the original metadata, so ReAct's already-rendered instructions hold.
        tools = getattr(program, "tools", None)
        if isinstance(tools, dict):
            for name, tool in list(tools.items()):
                if name in _LOCAL_TOOLS:
                    continue
                tools[name] = WorkflowTool(tool, program=call.program, options=options)

        lm = WorkflowLM(options=options)
        # track_usage=True so Module.acall accumulates per-call usage (fed by
        # WorkflowLM) and stamps it on the prediction. callbacks=[] so no spans
        # are emitted in workflow code -- the activities own span emission.
        with dspy.context(lm=lm, track_usage=True, callbacks=[]):
            prediction = await program.acall(**call.inputs)

        return ProgramCallOutput(
            prediction=prediction_to_dict(prediction),
            lm_usage=prediction.get_lm_usage() or None,
        )
