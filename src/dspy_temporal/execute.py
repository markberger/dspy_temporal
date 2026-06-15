"""Reusable workflow-side execution coroutines for both run modes.

These are the *bodies* of the two generic DSPy workflows, extracted so they can
be awaited from BOTH our generic ``@workflow.defn`` classes AND a user-authored
``@workflow.defn`` (composing a program into your own workflow via
``await agent.run(**inputs)``).

Because they run as workflow code, heavy/host imports follow the same
``imports_passed_through()`` discipline as ``fine/workflow.py``: the sandbox
reuses the already-imported host modules (most importantly the host program
registry) instead of reloading them and tripping its restriction checks. The
coroutines themselves stay replay-safe -- only ``workflow.execute_activity`` and
pure data shaping, no wall-clock or randomness.

``execute_coarse`` / ``execute_fine`` return a ``dspy.Prediction`` (what a
composing workflow wants). The generic workflows keep returning
``ProgramCallOutput`` over the wire: coarse via the shared ``_coarse_activity_call``
(single source of truth for the activity name + timeouts), fine by wrapping the
prediction at the workflow boundary.
"""

from __future__ import annotations

from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import dspy

    from .registry import all_named_predictors, default_registry
    from .serde import dict_to_prediction, normalize_inputs

from .models import LMDescribeInput, LMSpecsOutput, ProgramCallInput, ProgramCallOutput
from .options import CallOptions

# The fine seams (WorkflowLM / WorkflowTool) live in the ``fine`` package, whose
# ``__init__`` imports ``fine.workflow`` -- which imports *this* module. Importing
# them at module top would close that cycle (execute <-> fine.workflow) during
# import. They're only needed at call time, so ``execute_fine`` imports them
# lazily (below), keeping ``execute.py`` import-time independent of the fine
# package and the cycle impossible regardless of import order.

# Workflow-side activity-name references (the activities own the canonical
# @activity.defn names; these are the single place workflow code dispatches them).
ACTIVITY_NAME = "dspy_run_program"
DESCRIBE_ACTIVITY_NAME = "dspy_describe_lms"
DEFAULT_LM_REF = "__default__"

# ReAct's internal "finish" marker is a no-op (lambda: "Completed."); keep it
# local to the workflow instead of paying an activity round-trip for it.
_LOCAL_TOOLS = frozenset({"finish"})


async def _coarse_activity_call(
    call: ProgramCallInput, options: CallOptions, *, task_queue: str | None = None
) -> ProgramCallOutput:
    """Dispatch the single coarse program activity for an already-built call.

    Takes the ``ProgramCallInput`` and its ``CallOptions`` *separately*: the
    activity (``run_program_activity``) reads only ``call.program`` /
    ``call.inputs`` and ignores ``call.options``, while ``options`` drives the
    activity timeouts/retry here. Keeping options out of the payload at the
    construction site lets ``execute_coarse`` build the call with ``options=None``
    -- see its note on the sandbox ``CallOptions`` re-validation pitfall. Single
    source of truth for the activity name + timeouts/retry.

    ``task_queue`` routes the activity to a dedicated queue (the cheap-workflow-
    workers + dedicated-activity-pool split); ``None`` co-locates it with the
    calling workflow's queue (see ``CallOptions.activity_kwargs``).
    """
    return await workflow.execute_activity(
        ACTIVITY_NAME,
        call,
        result_type=ProgramCallOutput,
        **options.activity_kwargs(task_queue=task_queue),
    )


async def execute_coarse(
    name: str,
    inputs: dict[str, Any],
    options: CallOptions | None = None,
    *,
    task_queue: str | None = None,
) -> dspy.Prediction:
    """Run a coarse program from workflow code and return a ``dspy.Prediction``.

    Dispatches the whole program as one ``dspy_run_program`` activity and
    reconstructs a ``Prediction`` (with its ``lm_usage`` restored) for a composing
    workflow. ``normalize_inputs`` makes raw ``.run(**inputs)`` kwargs JSON-native
    before they cross the boundary. ``task_queue`` routes the activity to a
    dedicated queue (``None`` co-locates it with the calling workflow's queue).

    The ``ProgramCallInput`` is built with ``options=None`` on purpose: building it
    inside workflow/sandbox code with a *nested* ``CallOptions`` instance triggers
    a pydantic ``model_type`` validation against the sandbox's ``CallOptions``
    class object, which need not be identical to the instance's class (the classic
    Temporal-pydantic sandbox class-identity pitfall) and fails the workflow task.
    The activity ignores ``call.options`` anyway, so options live only as the
    timeouts/retry passed to ``_coarse_activity_call``.
    """
    options = options or CallOptions()
    call = ProgramCallInput(program=name, inputs=normalize_inputs(inputs))
    out = await _coarse_activity_call(call, options, task_queue=task_queue)
    return dict_to_prediction(out.prediction, out.lm_usage)


async def execute_fine(
    name: str,
    inputs: dict[str, Any],
    options: CallOptions | None = None,
    *,
    task_queue: str | None = None,
) -> dspy.Prediction:
    """Run a fine program from workflow code and return a ``dspy.Prediction``.

    Mirrors the coarse->fine split: the program's orchestration runs here in the
    workflow under a ``dspy.context`` that swaps in a per-predictor ``WorkflowLM``
    and activity-backed tools, so each LM call and tool call becomes its own
    activity (durable, independently retried) while the loop control stays
    deterministic. See ``fine/workflow.py`` for the full rationale.

    ``task_queue`` routes *every* per-call activity (the describe, and each
    ``dspy_lm_call`` / ``dspy_tool_call``) to a dedicated queue; ``None``
    co-locates them with the calling workflow's queue.
    """
    # Lazy (cycle-breaking) seam import; passthrough so the sandbox reuses host.
    with workflow.unsafe.imports_passed_through():
        from .fine.lm import WorkflowLM
        from .fine.tools import WorkflowTool

    options = options or CallOptions()

    # Describe each predictor's effective LM up front (one recorded activity ->
    # deterministic on replay). JSONAdapter / Predict read the LM's model,
    # capability flags, and kwargs *in the workflow* before the first call, so
    # WorkflowLM must carry them. Credentials stay on the worker.
    specs = await workflow.execute_activity(
        DESCRIBE_ACTIVITY_NAME,
        LMDescribeInput(program=name),
        result_type=LMSpecsOutput,
        **options.activity_kwargs(task_queue=task_queue),
    )

    program = default_registry().build(name)

    # Wrap each tool (ReAct and any module exposing a `.tools` dict) so its
    # execution dispatches to the dspy_tool_call activity. The wrapper keeps the
    # original metadata, so ReAct's already-rendered instructions hold.
    tools = getattr(program, "tools", None)
    if isinstance(tools, dict):
        for tool_name, tool in list(tools.items()):
            if tool_name in _LOCAL_TOOLS:
                continue
            tools[tool_name] = WorkflowTool(
                tool, program=name, options=options, task_queue=task_queue
            )

    # Bind a per-predictor WorkflowLM so each predictor routes to *its own* LM
    # (honoring a bound `.lm`); the activity resolves lm_ref -> real LM.
    # all_named_predictors (not named_predictors) so predictors inside a compiled
    # sub-module also get a WorkflowLM -- otherwise their bound `.lm` would win and
    # call the real LM inside the sandbox instead of the dspy_lm_call activity.
    default_spec = specs.specs[DEFAULT_LM_REF]
    for predictor_name, predictor in all_named_predictors(program):
        predictor.lm = WorkflowLM(
            spec=specs.specs.get(predictor_name) or default_spec,
            lm_ref=predictor_name,
            program=name,
            options=options,
            task_queue=task_queue,
        )
    # Context fallback for any predictor created dynamically at call time (not in
    # the startup walk) -> routes to the worker default LM.
    default_lm = WorkflowLM(
        spec=default_spec,
        lm_ref=DEFAULT_LM_REF,
        program=name,
        options=options,
        task_queue=task_queue,
    )
    # track_usage=True so Module.acall accumulates per-call usage (fed by
    # WorkflowLM) and stamps it on the prediction. callbacks=[] so no spans are
    # emitted in workflow code -- the activities own span emission.
    with dspy.context(lm=default_lm, track_usage=True, callbacks=[]):
        return await program.acall(**inputs)
