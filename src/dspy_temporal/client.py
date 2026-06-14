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
    and is the *primary* path (the handle carries mode + queue so they can't
    desync). ``task_queue`` is required (no default).

    The run ``mode`` is **resolved from the registry**, not blindly trusted:

    - If ``name`` was deployed in this process with a mode, that mode is used. A
      conflicting explicit ``mode`` raises (use ``handle.start`` to avoid the
      mismatch entirely, or pass the matching mode / omit it).
    - If ``name`` was registered *without* a mode (via the low-level
      ``register_program``), an explicit ``mode`` is required (none -> raises).
    - If ``name`` is not registered in this process (a thin client that never
      imported the program module), an explicit ``mode`` is required as the
      escape hatch (none -> raises, since the mode would be ambiguous).
    """
    reg = default_registry()
    if name in reg:
        registered = reg.mode_for(name)
        if registered is None:  # registered locally but no mode (register_program)
            if mode is None:
                raise ValueError(
                    f"Program {name!r} is registered without a run mode and no mode "
                    f"was given. Pass mode=RunMode.COARSE/FINE, or deploy() it with a "
                    f"mode."
                )
            resolved = mode
        else:
            if mode is not None and mode != registered:
                raise ValueError(
                    f"Program {name!r} is registered as mode={registered.value!r} but "
                    f"run_program was called with mode={mode.value!r}. Use "
                    f"handle.start() (the can't-desync path), or pass the matching "
                    f"mode / omit it."
                )
            resolved = registered
    else:  # not registered in this process
        if mode is None:
            raise ValueError(
                f"Program {name!r} is not registered in this process and no mode was "
                f"given -> ambiguous. Import the program module here, or pass "
                f"mode=RunMode.COARSE/FINE (thin-client escape hatch)."
            )
        resolved = mode

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
