"""Auto-wrap API: register a dspy.Module (builder or instance) and run it on Temporal.

``deploy(source, *, name, task_queue, mode=...)`` accepts a live ``dspy.Module``
instance *or* a zero-arg builder and returns a :class:`TemporalProgram` handle
that carries the program's ``mode`` + ``task_queue``.

The handle is the single source of truth for how the program runs:

- ``await handle.run(**inputs)`` -- inside a user-authored ``@workflow.defn`` it
  dispatches our activities inline (compose a deployed program into your own
  workflow); outside any workflow it degrades to a plain local DSPy call.
- ``await handle.start(client, **inputs)`` -- start the program as a standalone
  workflow from a client, using the handle's own ``mode`` + ``task_queue``.

``dspy_temporal.run_program(client, name, inputs, *, task_queue, ...)`` is the
low-level by-name escape hatch ``start`` delegates to.
"""

from __future__ import annotations

from dataclasses import dataclass

import dspy
from temporalio import workflow

from ..config import RunMode, run_program_async_or_sync
from ..execute import execute_coarse, execute_fine
from ..fine.workflow import DSPyProgramFineWorkflow
from ..options import CallOptions
from ..registry import ModuleSource, default_registry, register_program
from .workflow import DSPyProgramWorkflow


def _workflow_run_for_mode(mode: RunMode):
    """Pick the workflow entrypoint for a run mode (both take ProgramCallInput).

    ``TemporalProgram`` is mode-agnostic; it just branches here rather than
    moving out of the coarse package (avoids a churny re-home).
    """
    return (
        DSPyProgramFineWorkflow.run if mode == RunMode.FINE else DSPyProgramWorkflow.run
    )


@dataclass
class TemporalProgram:
    """Handle returned by :func:`deploy`.

    Carries the program's ``task_queue`` and ``mode`` so the handle alone knows
    *where* and *how* to run: ``mode`` selects coarse (whole-program activity)
    vs. fine (per-call activities), and ``task_queue`` is the queue ``start``
    dispatches to (the same one a serving worker is built on).
    """

    name: str
    # Required: must precede the defaulted ``mode`` (a non-default dataclass
    # field can't follow a defaulted one).
    task_queue: str
    mode: RunMode = RunMode.COARSE

    async def run(self, **inputs) -> dspy.Prediction:
        """Run the program, dispatching by execution context.

        Inside a workflow (a user's own ``@workflow.defn`` that awaits this) the
        call dispatches our activities inline via ``execute_coarse`` /
        ``execute_fine``. Outside any workflow it degrades to a plain in-process
        DSPy call against the locally configured LM (no worker-LM injection --
        ``start`` is the path that uses the worker).
        """
        opts = None
        if workflow.in_workflow():
            if self.mode == RunMode.FINE:
                return await execute_fine(self.name, inputs, opts)
            return await execute_coarse(self.name, inputs, opts)
        # In-process degrade: build from the registry and run locally.
        program = default_registry().build(self.name)
        if self.mode == RunMode.FINE:
            return await program.acall(**inputs)
        # Coarse: prefer the async path (so concurrent sub-calls trace correctly),
        # falling back to the sync call for forward-only modules.
        return await run_program_async_or_sync(program, inputs)

    async def start(
        self,
        client,
        *,
        workflow_id: str | None = None,
        options: CallOptions | None = None,
        **inputs,
    ) -> dspy.Prediction:
        """Start this program as a standalone workflow and await its result.

        Uses the handle's own ``mode`` + ``task_queue`` -- the handle is the
        single source of truth, so the caller never re-passes them. Delegates to
        :func:`dspy_temporal.run_program` (the by-name escape hatch).
        """
        # Lazy import: client.py imports ``_workflow_run_for_mode`` from this
        # module at top level, so a top-level back-import would form a cycle.
        from ..client import run_program

        return await run_program(
            client,
            self.name,
            inputs,
            task_queue=self.task_queue,
            workflow_id=workflow_id,
            options=options,
            mode=self.mode,
        )


def deploy(
    source: ModuleSource,
    *,
    name: str,
    task_queue: str,
    mode: RunMode = RunMode.COARSE,
) -> TemporalProgram:
    """Register a ``dspy.Module`` instance *or* a builder and return a handle.

    Accepts a live ``dspy.Module`` (e.g. a compiled program with few-shot demos:
    its prototype stays in worker memory and each run gets a fresh, LM-stripped
    clone) or a zero-arg builder. ``task_queue`` is required (no default); the
    returned handle carries it along with ``mode``.
    """
    register_program(name, source)
    return TemporalProgram(name=name, task_queue=task_queue, mode=mode)
