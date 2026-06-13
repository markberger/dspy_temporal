"""Client-side helper to start a program workflow and get a Prediction back."""

from __future__ import annotations

import uuid

from temporalio.client import Client

from .coarse.api import _workflow_run_for_mode
from .config import CallOptions
from .models import ProgramCallInput
from .serde import dict_to_prediction, normalize_inputs


async def run_program(
    client: Client,
    name: str,
    inputs: dict,
    *,
    task_queue: str = "dspy-temporal",
    workflow_id: str | None = None,
    options: CallOptions | None = None,
    mode: str = "coarse",
):
    """Start the program workflow, wait for it, and return a ``dspy.Prediction``.

    ``mode`` picks the workflow: ``"coarse"`` runs the whole program in one
    activity; ``"fine"`` orchestrates per-LM-call / per-tool-call activities.
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
