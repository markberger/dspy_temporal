"""Client-side helpers to start a program workflow and get a Prediction back.

Two shapes, sharing one path:

- ``run_program`` -- start *and* await the result (blocking).
- ``start_program_nowait`` -- start and hand back the :class:`WorkflowHandle`
  immediately, so a caller (a web request, a dashboard) can poll
  ``handle.describe()`` / ``handle.result()`` on its own schedule. Pair it with
  ``prediction_of(handle)`` to decode that handle's wire result back into a
  ``dspy.Prediction``.

``run_program`` is just ``start_program_nowait`` followed by
``prediction_of`` -- the same factoring Temporal's own ``execute_workflow`` is
(``start_workflow`` + ``WorkflowHandle.result``).
"""

from __future__ import annotations

import uuid

from temporalio.client import Client, WorkflowHandle

from .coarse.workflow import DSPyProgramWorkflow
from .config import CallOptions, RunMode
from .fine.workflow import DSPyProgramFineWorkflow
from .models import ProgramCallInput, ProgramCallOutput
from .registry import default_registry
from .serde import dict_to_prediction, normalize_inputs


def _workflow_run_for_mode(mode: RunMode):
    """Pick the workflow entrypoint for a run mode (both take ProgramCallInput)."""
    return (
        DSPyProgramFineWorkflow.run if mode == RunMode.FINE else DSPyProgramWorkflow.run
    )


async def start_program_nowait(
    client: Client,
    name: str,
    inputs: dict,
    *,
    task_queue: str,
    workflow_id: str | None = None,
    options: CallOptions | None = None,
    mode: RunMode | None = None,
) -> WorkflowHandle:
    """Start the program workflow and return its handle **without** awaiting it.

    The non-blocking sibling of :func:`run_program`: identical mode resolution
    and input handling, but it calls ``client.start_workflow(...)`` and hands the
    :class:`WorkflowHandle` straight back, so the caller drives
    ``handle.describe()`` / ``await handle.result()`` on its own schedule (the
    start-now / poll-later shape). Pair it with :func:`prediction_of` (or
    ``ref.result_of``) to decode the result into a ``dspy.Prediction`` /
    typed value.

    The handle is started over the generic ``DSPyProgram[Fine]Workflow``, whose
    ``-> ProgramCallOutput`` return annotation Temporal uses to type the handle,
    so ``await handle.result()`` decodes straight into a ``ProgramCallOutput``.

    The run ``mode`` is **resolved from the registry** (see
    :meth:`ProgramRegistry.resolve_mode`), not blindly trusted:

    - If ``name`` was registered in this process with a mode, that mode is used. A
      conflicting explicit ``mode`` raises (use ``ref.start_nowait`` to avoid the
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
    return await client.start_workflow(
        _workflow_run_for_mode(resolved),
        call,
        id=workflow_id or f"dspy-{name}-{uuid.uuid4().hex[:12]}",
        task_queue=task_queue,
    )


async def prediction_of(handle: WorkflowHandle):
    """Await a handle from :func:`start_program_nowait` and rebuild the Prediction.

    Decodes the workflow's ``ProgramCallOutput`` wire result back into a
    ``dspy.Prediction`` (restoring ``lm_usage``). Tolerant of a handle that
    carries no result type -- e.g. one re-obtained across requests via
    ``client.get_workflow_handle(workflow_id)``, where Temporal hands back the
    raw decoded dict -- by validating it into ``ProgramCallOutput`` either way.
    """
    output = await handle.result()
    if not isinstance(output, ProgramCallOutput):
        output = ProgramCallOutput.model_validate(output)
    return dict_to_prediction(output.prediction, output.lm_usage)


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
    desync). ``task_queue`` is required (no default). For the non-blocking shape
    (start now, poll later), use :func:`start_program_nowait` + :func:`prediction_of`
    -- this is exactly that pair, awaited in one shot. Mode resolution is
    delegated to :func:`start_program_nowait`.
    """
    handle = await start_program_nowait(
        client,
        name,
        inputs,
        task_queue=task_queue,
        workflow_id=workflow_id,
        options=options,
        mode=mode,
    )
    return await prediction_of(handle)
