"""Auto-wrap API: register a dspy.Module (builder or instance) and run it on Temporal.

``deploy(source, name=..., mode=..., task_queue=...)`` accepts a live
``dspy.Module`` instance *or* a zero-arg builder and returns a
:class:`TemporalProgram` handle.

The handle runs the program in a context-aware way:

- ``await handle.run(**inputs)`` -- inside a user-authored ``@workflow.defn`` it
  dispatches our activities inline (compose a deployed program into your own
  workflow); outside any workflow it degrades to a plain local DSPy call.

To start a deployed program as a standalone workflow from a client, use
``dspy_temporal.run_program(client, name, inputs, ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import dspy
from temporalio import workflow

from ..config import RunConfig, RunMode, run_program_async_or_sync
from ..execute import execute_coarse, execute_fine
from ..fine.workflow import DSPyProgramFineWorkflow
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

    Mode-agnostic: ``self.config.mode`` selects coarse (whole-program activity)
    vs. fine (per-call activities) wherever the program runs.
    """

    name: str
    config: RunConfig

    async def run(self, **inputs) -> dspy.Prediction:
        """Run the program, dispatching by execution context.

        Inside a workflow (a user's own ``@workflow.defn`` that awaits this) the
        call dispatches our activities inline via ``execute_coarse`` /
        ``execute_fine``. Outside any workflow it degrades to a plain in-process
        DSPy call against the locally configured LM (no worker-LM injection --
        ``run_program`` is the path that uses the worker).
        """
        opts = None
        if workflow.in_workflow():
            if self.config.mode == RunMode.FINE:
                return await execute_fine(self.name, inputs, opts)
            return await execute_coarse(self.name, inputs, opts)
        # In-process degrade: build from the registry and run locally.
        program = default_registry().build(self.name)
        if self.config.mode == RunMode.FINE:
            return await program.acall(**inputs)
        # Coarse: prefer the async path (so concurrent sub-calls trace correctly),
        # falling back to the sync call for forward-only modules.
        return await run_program_async_or_sync(program, inputs)


def deploy(
    source: ModuleSource,
    *,
    name: str,
    mode: RunMode = RunMode.COARSE,
    task_queue: str = "dspy-temporal",
    config: RunConfig | None = None,
) -> TemporalProgram:
    """Register a ``dspy.Module`` instance *or* a builder and return a handle.

    Accepts a live ``dspy.Module`` (e.g. a compiled program with few-shot demos:
    its prototype stays in worker memory and each run gets a fresh, LM-stripped
    clone) or a zero-arg builder. When ``config`` is omitted it is assembled from
    ``mode`` + ``task_queue``; a supplied ``config`` takes precedence (it carries
    its own mode/task_queue).
    """
    register_program(name, source)
    if config is None:
        config = RunConfig(task_queue=task_queue, mode=mode)
    return TemporalProgram(name=name, config=config)
