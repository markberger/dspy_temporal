"""The single coarse-mode activity: run a whole DSPy program.

This runs outside the workflow sandbox, so DSPy is fully intact -- its adapters,
caching, retries, and LM calls all work normally. One activity == one program
run; durability is at the job level (a crash re-runs the whole program).
"""

from __future__ import annotations

import contextlib

import dspy
from temporalio import activity

from ..config import get_tracing_callback, get_worker_lm, run_program_async_or_sync
from ..heartbeat import heartbeating
from ..models import ProgramCallInput, ProgramCallOutput
from ..registry import default_registry
from ..serde import prediction_to_dict


@activity.defn(name="dspy_run_program")
async def run_program_activity(call: ProgramCallInput) -> ProgramCallOutput:
    registry = default_registry()
    program = registry.build(call.program)

    # Apply the worker LM (and usage tracking) as a thread-local override so the
    # program's predictors that don't carry their own LM use it. A predictor's
    # own .lm still takes precedence (see Predict._forward_preprocess).
    ctx_kwargs = {"track_usage": True}
    worker_lm = get_worker_lm()
    if worker_lm is not None:
        ctx_kwargs["lm"] = worker_lm
    # Attach the tracing callback (if tracing is set up) so DSPy emits spans for
    # this run. Span emission lives here, inside the activity (never in workflow
    # code), so it is replay-safe by construction.
    tracing_callback = get_tracing_callback()
    if tracing_callback is not None:
        callbacks = list(dspy.settings.callbacks or [])
        # Guard against a double-add: a user may also have registered this same
        # callback globally via dspy.settings.callbacks, which would double-emit
        # every span.
        if tracing_callback not in callbacks:
            callbacks.append(tracing_callback)
        ctx_kwargs["callbacks"] = callbacks

    # Prefer DSPy's async path (program.acall) so in-program concurrency done with
    # asyncio.gather propagates ACTIVE_CALL_ID and tracing spans nest correctly;
    # sync-only modules fall back to program(). A background thread heartbeats while
    # the program runs, so a configured heartbeat_timeout keeps the activity alive
    # instead of timing it out. No-op when no heartbeat_timeout is set.
    with heartbeating(), dspy.context(**ctx_kwargs):
        prediction = await run_program_async_or_sync(program, call.inputs)

    lm_usage = None
    with contextlib.suppress(Exception):
        usage = prediction.get_lm_usage()
        lm_usage = usage or None

    return ProgramCallOutput(
        prediction=prediction_to_dict(prediction),
        lm_usage=lm_usage,
    )
