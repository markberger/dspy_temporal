"""Auto-wrap API: register a dspy.Module (builder or instance) and run it on Temporal.

Three ways to get a runnable handle:

- ``deploy_module(name, builder)`` -- the original builder-oriented entry point.
- ``deploy(source, name=..., mode=..., task_queue=...)`` -- accepts a live
  ``dspy.Module`` instance *or* a zero-arg builder, with run config assembled from
  keywords.

Both return a :class:`TemporalProgram` handle, which runs the program three ways:

- ``await handle.run(**inputs)`` -- context-aware: inside a user-authored
  ``@workflow.defn`` it dispatches our activities inline (compose a deployed
  program into your own workflow); outside any workflow it degrades to a plain
  local DSPy call.
- ``await handle.start(client, inputs, ...)`` -- the standalone path: start the
  generic program workflow on a client and await it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import dspy
from temporalio import workflow
from temporalio.client import Client

from ..config import CallOptions, RunConfig, RunMode, run_program_async_or_sync
from ..execute import execute_coarse, execute_fine
from ..fine.workflow import DSPyProgramFineWorkflow
from ..models import ProgramCallInput
from ..registry import ModuleSource, default_registry, register_program
from ..serde import dict_to_prediction, normalize_inputs
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
    """Handle returned by :func:`deploy` / :func:`deploy_module`.

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
        ``start`` is the path that uses the worker).
        """
        opts = self.config.call_options
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

    async def start(
        self,
        client: Client,
        inputs: dict,
        *,
        workflow_id: str | None = None,
        task_queue: str | None = None,
        options: CallOptions | None = None,
    ) -> dspy.Prediction:
        """Start the program as a standalone workflow and return a ``dspy.Prediction``.

        The workflow is selected by ``self.config.mode`` (``RunMode.COARSE`` ->
        whole-program activity; ``RunMode.FINE`` -> per-call activities).
        """
        call = ProgramCallInput(
            program=self.name,
            inputs=normalize_inputs(inputs),
            options=options or self.config.call_options,
        )
        output = await client.execute_workflow(
            _workflow_run_for_mode(self.config.mode),
            call,
            id=workflow_id or f"dspy-{self.name}-{uuid.uuid4().hex[:12]}",
            task_queue=task_queue or self.config.task_queue,
        )
        return dict_to_prediction(output.prediction, output.lm_usage)


def deploy_module(
    name: str,
    builder: ModuleSource,
    *,
    config: RunConfig | None = None,
) -> TemporalProgram:
    """Register a zero-arg ``dspy.Module`` builder under ``name``.

    ``builder`` is normally a callable returning a fresh ``dspy.Module`` (e.g.
    ``lambda: dspy.ChainOfThought("question -> answer")``), so no live LM or API
    key is ever serialized into Temporal history. A live ``dspy.Module`` instance
    is also accepted (it is cloned LM-stripped per run -- see :func:`deploy` for
    the instance-oriented entry point).
    """
    register_program(name, builder)
    return TemporalProgram(name=name, config=config or RunConfig())


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
