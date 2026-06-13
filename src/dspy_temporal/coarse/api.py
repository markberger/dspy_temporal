"""Auto-wrap API: register a dspy.Module builder and run it on Temporal."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from temporalio.client import Client

from ..config import CallOptions, RunConfig
from ..fine.workflow import DSPyProgramFineWorkflow
from ..models import ProgramCallInput
from ..registry import ModuleBuilder, register_program
from ..serde import dict_to_prediction, normalize_inputs
from .workflow import DSPyProgramWorkflow


def _workflow_run_for_mode(mode: str):
    """Pick the workflow entrypoint for a run mode (both take ProgramCallInput).

    ``DeployedProgram`` is mode-agnostic; it just branches here rather than
    moving out of the coarse package (avoids a churny re-home).
    """
    return DSPyProgramFineWorkflow.run if mode == "fine" else DSPyProgramWorkflow.run


@dataclass
class DeployedProgram:
    """Handle returned by :func:`deploy_module`."""

    name: str
    config: RunConfig

    async def execute(
        self,
        client: Client,
        inputs: dict,
        *,
        workflow_id: str | None = None,
        task_queue: str | None = None,
        options: CallOptions | None = None,
    ):
        """Run the program as a workflow and return a ``dspy.Prediction``.

        The workflow is selected by ``self.config.mode`` (``"coarse"`` ->
        whole-program activity; ``"fine"`` -> per-call activities).
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
    builder: ModuleBuilder,
    *,
    config: RunConfig | None = None,
) -> DeployedProgram:
    """Register a zero-arg ``dspy.Module`` builder under ``name``.

    ``builder`` must be a callable returning a fresh ``dspy.Module`` (e.g.
    ``lambda: dspy.ChainOfThought("question -> answer")``), so no live LM or API
    key is ever serialized into Temporal history.
    """
    register_program(name, builder)
    return DeployedProgram(name=name, config=config or RunConfig())
