"""Compose a deployed DSPy program inside your OWN Temporal workflow.

``agent.run(**inputs)`` works *inside* a user-authored ``@workflow.defn``: it
dispatches our activities inline, so you can interleave DSPy calls with your own
workflow logic (timers, other activities, child workflows) and the whole thing is
one durable, replayable execution. Outside a workflow the same call degrades to a
plain local DSPy call.

This file is the *workflow file*: Temporal's sandbox re-execs it on every
workflow task for deterministic isolation, so it must stay free of import-time
side effects. The ``deploy`` that registers ``"compose_qa"`` therefore lives in
``compose_agents.py``; we passthrough-import the handle so the sandbox reuses the
worker process's already-deployed program instead of re-running ``deploy`` (which
would rebuild and re-register the program on every task).

Register the user workflow on the worker via ``build_worker(...,
extra_workflows=[ResearchWorkflow])`` or ``DSPyPlugin(extra_workflows=[...])``.
Imported by ``examples/worker.py`` (serves the workflow) and
``examples/run_compose.py`` (starts it).
"""

from datetime import timedelta

from temporalio import workflow

# Passthrough so the sandbox references the worker process's already-loaded
# module (and its one-time deploy) instead of re-executing it each task. Outside
# the sandbox (a normal host import) this context manager is a no-op.
with workflow.unsafe.imports_passed_through():
    from compose_agents import (  # noqa: F401  (TASK_QUEUE re-exported for run_compose.py)
        TASK_QUEUE,
        triage_agent,
    )


@workflow.defn(name="ResearchWorkflow")
class ResearchWorkflow:
    """A user workflow that calls a deployed DSPy program as durable steps."""

    @workflow.run
    async def run(self, question: str) -> str:
        # Each agent.run dispatches the program's activity inline -- a durable,
        # retried step recorded in THIS workflow's history.
        first = await triage_agent.run(question=question)
        # Interleave ordinary workflow logic (here a timer) between DSPy calls.
        await workflow.sleep(timedelta(seconds=1))
        followup = await triage_agent.run(
            question=f"In one word, summarize this answer: {first.answer}"
        )
        return followup.answer
