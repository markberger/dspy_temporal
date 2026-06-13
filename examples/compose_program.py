"""Win B: compose a deployed DSPy program inside your OWN Temporal workflow.

``agent.run(**inputs)`` works *inside* a user-authored ``@workflow.defn``: it
dispatches our activities inline, so you can interleave DSPy calls with your own
workflow logic (timers, other activities, child workflows) and the whole thing is
one durable, replayable execution. Outside a workflow the same call degrades to a
plain local DSPy call.

Register the user workflow on the worker via ``build_worker(...,
extra_workflows=[ResearchWorkflow])`` or ``DSPyPlugin(extra_workflows=[...])``.
Imported by ``examples/worker.py`` (serves the workflow) and
``examples/run_compose.py`` (starts it).
"""

from datetime import timedelta

import dspy
from temporalio import workflow

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"

# A deployed program, composed into the workflow below via agent.run().
triage_agent = dt.deploy(
    lambda: dspy.ChainOfThought("question -> answer"),
    name="compose_qa",
    mode=dt.RunMode.COARSE,
    task_queue=TASK_QUEUE,
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
