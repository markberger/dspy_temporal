"""Worker construction: register both workflows + all activities.

The fixed activity/workflow set lives in :mod:`dspy_temporal.plugin`
(:data:`DSPY_ACTIVITIES` / :data:`DSPY_WORKFLOWS`) so ``build_worker`` and
:class:`DSPyPlugin` share a single source of truth. ``build_worker`` wires the
``Worker`` directly (it does *not* apply the plugin, to avoid double-application)
but pulls the same constants.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from .config import RunConfig
from .plugin import DSPY_ACTIVITIES, DSPY_WORKFLOWS
from .sandbox import default_workflow_runner


def build_worker(
    client: Client,
    *,
    config: RunConfig | None = None,
    max_concurrent_activities: int = 100,
    extra_passthrough_modules: tuple[str, ...] = (),
    extra_workflows: tuple = (),
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

    ``extra_workflows`` registers additional ``@workflow.defn`` classes on the
    same worker -- e.g. a user-authored workflow that composes a deployed program
    via ``await agent.run(**inputs)`` (see ``TemporalProgram.run``).
    """
    config = config or RunConfig()
    worker_kwargs.setdefault(
        "workflow_runner", default_workflow_runner(*extra_passthrough_modules)
    )
    return Worker(
        client,
        task_queue=config.task_queue,
        workflows=[*DSPY_WORKFLOWS, *extra_workflows],
        activities=list(DSPY_ACTIVITIES),
        activity_executor=ThreadPoolExecutor(max_workers=max_concurrent_activities),
        **worker_kwargs,
    )
