"""Client-side helper to start a program workflow and get a Prediction back."""

from __future__ import annotations

import uuid

from temporalio.client import Client

from .coarse.workflow import DSPyProgramWorkflow
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
):
    """Start the program workflow, wait for it, and return a ``dspy.Prediction``."""
    call = ProgramCallInput(
        program=name,
        inputs=normalize_inputs(inputs),
        options=options,
    )
    output = await client.execute_workflow(
        DSPyProgramWorkflow.run,
        call,
        id=workflow_id or f"dspy-{name}-{uuid.uuid4().hex[:12]}",
        task_queue=task_queue,
    )
    return dict_to_prediction(output.prediction)
