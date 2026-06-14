"""Worker construction: delegate the DSPy worker set to ``DSPyPlugin``.

``build_worker`` constructs its ``Worker`` *through* :class:`DSPyPlugin`, so the
plugin is the single source of truth for the activities, workflows, sandbox
runner, and activity executor (the fixed set lives in
:mod:`dspy_temporal.plugin` as :data:`DSPY_ACTIVITIES` / :data:`DSPY_WORKFLOWS`).

It expects a plugin-free client (the usual ``dt.connect("localhost:7233")``
flow). If the client already carries a ``DSPyPlugin`` (e.g.
``dt.connect(..., plugins=[DSPyPlugin()])``), that plugin already propagates to
the worker -- pass the client to a bare ``Worker`` in that case rather than
``build_worker``, or Temporal will warn that the same plugin type is applied twice.
"""

from __future__ import annotations

from temporalio.client import Client
from temporalio.worker import Worker

from .config import RunConfig
from .plugin import DSPyPlugin


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

    The activities/workflows/runner/executor are all contributed by
    :class:`DSPyPlugin`, which this builds the ``Worker`` through.
    """
    config = config or RunConfig()
    plugin = DSPyPlugin(
        extra_passthrough_modules=extra_passthrough_modules,
        max_concurrent_activities=max_concurrent_activities,
        extra_workflows=extra_workflows,
    )
    plugins = [*worker_kwargs.pop("plugins", ()), plugin]
    return Worker(
        client,
        task_queue=config.task_queue,
        plugins=plugins,
        **worker_kwargs,
    )
