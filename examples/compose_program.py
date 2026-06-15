"""Compose a program inside your OWN Temporal workflow.

``triage_agent.run(**inputs)`` works *inside* a user-authored ``@workflow.defn``:
it dispatches our activities inline, so you can interleave DSPy calls with your own
workflow logic (timers, other activities, child workflows) and the whole thing is
one durable, replayable execution. Outside a workflow the same call degrades to a
plain local DSPy call.

This is the *workflow file*. It imports the program reference with a **plain
import** -- no ``imports_passed_through()`` dance -- because ``compose_refs`` is
side-effect-free (``dt.program(...)`` registers nothing and loads no model). The
implementation is bound on the worker (``examples/worker.py``); this file never
touches dspy.

Register the user workflow on the worker via ``build_worker(...,
extra_workflows=[ResearchWorkflow])`` or ``DSPyPlugin(extra_workflows=[...])``.
Imported by ``examples/worker.py`` (serves the workflow) and
``examples/run_compose.py`` (starts it).
"""

from datetime import timedelta

# Plain import: compose_refs has zero import-time side effects, so the workflow
# file needs no passthrough block and never risks the registry sandbox guardrail.
from compose_refs import (  # noqa: F401  (TASK_QUEUE re-exported for run_compose.py)
    TASK_QUEUE,
    Answer,
    triage_agent,
)
from temporalio import workflow


@workflow.defn(name="ResearchWorkflow")
class ResearchWorkflow:
    """A user workflow that calls a program reference as durable steps."""

    @workflow.run
    async def run(self, question: str) -> Answer:
        # Each triage_agent.run dispatches the program's activity inline -- a
        # durable, retried step recorded in THIS workflow's history. The ref's
        # ``result`` adapter hands back a typed ``Answer``, so dspy never enters
        # this workflow code.
        first = await triage_agent.run(question=question)
        # Interleave ordinary workflow logic (here a timer) between DSPy calls.
        await workflow.sleep(timedelta(seconds=1))
        return await triage_agent.run(
            question=f"In one word, summarize this answer: {first.text}"
        )
