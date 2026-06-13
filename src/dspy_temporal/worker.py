"""Worker construction: register both workflows + all activities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from .coarse.activities import run_program_activity
from .coarse.workflow import DSPyProgramWorkflow
from .config import RunConfig
from .fine.activities import describe_lms_activity, lm_call_activity, tool_call_activity
from .fine.workflow import DSPyProgramFineWorkflow
from .sandbox import default_workflow_runner


def build_worker(
    client: Client,
    *,
    config: RunConfig | None = None,
    max_concurrent_activities: int = 100,
    extra_passthrough_modules: tuple[str, ...] = (),
    **worker_kwargs,
) -> Worker:
    """Build a Temporal ``Worker`` that serves all registered DSPy programs.

    One worker serves both modes: it registers the coarse and fine workflows and
    every activity. All activities are synchronous (they run blocking DSPy/LM
    calls), so a shared ``ThreadPoolExecutor`` backs them; fine mode adds no new
    executor. The workflow runner passes dspy and its I/O deps through the
    sandbox (they are activity-only, or run as passthrough in the fine workflow).

    ``extra_passthrough_modules`` is the fine-mode escape hatch: extra module
    prefixes to share from the host, for the rare builder that references a
    module whose import-time side effects would trip the sandbox.
    """
    config = config or RunConfig()
    worker_kwargs.setdefault(
        "workflow_runner", default_workflow_runner(*extra_passthrough_modules)
    )
    return Worker(
        client,
        task_queue=config.task_queue,
        workflows=[DSPyProgramWorkflow, DSPyProgramFineWorkflow],
        activities=[
            run_program_activity,
            describe_lms_activity,
            lm_call_activity,
            tool_call_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=max_concurrent_activities),
        **worker_kwargs,
    )
