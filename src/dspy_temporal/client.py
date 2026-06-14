"""Client-side helper to start a program workflow and get a Prediction back."""

from __future__ import annotations

import uuid

from temporalio.client import Client

from .coarse.workflow import DSPyProgramWorkflow
from .config import CallOptions, RunMode
from .fine.workflow import DSPyProgramFineWorkflow
from .models import ProgramCallInput
from .serde import dict_to_prediction, normalize_inputs


def _workflow_run_for_mode(mode: RunMode):
    """Pick the workflow entrypoint for a run mode (both take ProgramCallInput)."""
    return (
        DSPyProgramFineWorkflow.run if mode == RunMode.FINE else DSPyProgramWorkflow.run
    )


async def run_program(
    client: Client,
    name: str,
    inputs: dict,
    *,
    task_queue: str,
    workflow_id: str | None = None,
    options: CallOptions | None = None,
    mode: RunMode = RunMode.COARSE,
):
    """Start the program workflow, wait for it, and return a ``dspy.Prediction``.

    The low-level by-name escape hatch (``TemporalProgram.start`` delegates here).
    ``task_queue`` is required (no default). ``mode`` picks the workflow:
    ``RunMode.COARSE`` runs the whole program in one activity; ``RunMode.FINE``
    orchestrates per-LM-call / per-tool-call activities.
    """
    call = ProgramCallInput(
        program=name,
        inputs=normalize_inputs(inputs),
        options=options,
    )
    output = await client.execute_workflow(
        _workflow_run_for_mode(mode),
        call,
        id=workflow_id or f"dspy-{name}-{uuid.uuid4().hex[:12]}",
        task_queue=task_queue,
    )
    return dict_to_prediction(output.prediction, output.lm_usage)
