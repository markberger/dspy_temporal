"""Client-side helper to start a program workflow and get a Prediction back."""

from __future__ import annotations

import uuid

from temporalio.client import Client

from .coarse.workflow import DSPyProgramWorkflow
from .config import CallOptions, RunMode
from .fine.workflow import DSPyProgramFineWorkflow
from .models import ProgramCallInput
from .registry import default_registry
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
    mode: RunMode | None = None,
):
    """Start the program workflow, wait for it, and return a ``dspy.Prediction``.

    The low-level by-name escape hatch; ``TemporalProgram.start`` delegates here
    and is the *primary* path (the ref carries mode + queue so they can't
    desync). ``task_queue`` is required (no default).

    The run ``mode`` is **resolved from the registry** (see
    :meth:`ProgramRegistry.resolve_mode`), not blindly trusted:

    - If ``name`` was registered in this process with a mode, that mode is used. A
      conflicting explicit ``mode`` raises (use ``ref.start`` to avoid the
      mismatch entirely, or pass the matching mode / omit it).
    - If ``name`` was registered *without* a mode (via the low-level
      ``register_program``), an explicit ``mode`` is required (none -> raises).
    - If ``name`` is not registered in this process (a thin client that never
      imported the program module), an explicit ``mode`` is required as the
      escape hatch (none -> raises, since the mode would be ambiguous).
    """
    resolved = default_registry().resolve_mode(name, mode)

    call = ProgramCallInput(
        program=name,
        inputs=normalize_inputs(inputs),
        options=options,
    )
    output = await client.execute_workflow(
        _workflow_run_for_mode(resolved),
        call,
        id=workflow_id or f"dspy-{name}-{uuid.uuid4().hex[:12]}",
        task_queue=task_queue,
    )
    return dict_to_prediction(output.prediction, output.lm_usage)
