"""The single coarse-mode activity: run a whole DSPy program.

This runs outside the workflow sandbox, so DSPy is fully intact -- its adapters,
caching, retries, and LM calls all work normally. One activity == one program
run; durability is at the job level (a crash re-runs the whole program).
"""

from __future__ import annotations

import contextlib

import dspy
from temporalio import activity

from ..config import get_worker_lm
from ..models import ProgramCallInput, ProgramCallOutput
from ..registry import default_registry
from ..serde import prediction_to_dict


@activity.defn(name="dspy_run_program")
def run_program_activity(call: ProgramCallInput) -> ProgramCallOutput:
    registry = default_registry()
    program = registry.build(call.program)

    worker_lm = get_worker_lm()
    # Apply the worker LM (and usage tracking) as a thread-local override so the
    # program's predictors that don't carry their own LM use it. A predictor's
    # own .lm still takes precedence (see Predict._forward_preprocess).
    ctx = (
        dspy.context(lm=worker_lm, track_usage=True)
        if worker_lm is not None
        else dspy.context(track_usage=True)
    )
    with ctx:
        prediction = program(**call.inputs)

    lm_usage = None
    with contextlib.suppress(Exception):
        usage = prediction.get_lm_usage()
        lm_usage = usage or None

    return ProgramCallOutput(
        prediction=prediction_to_dict(prediction),
        lm_usage=lm_usage,
    )
