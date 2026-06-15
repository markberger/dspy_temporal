"""End-to-end: a user @workflow.defn composes a program reference.

Proves ``await triage_agent.run(**inputs)`` works *inside* a user-authored
workflow: ``TemporalProgram.run`` sees it is in a workflow and dispatches
``execute_coarse`` (our program activity) inline. The user workflow
(``ResearchWorkflow`` from examples/compose_program.py) chains two such calls
across a ``workflow.sleep`` and is served on the same worker via
``build_worker(extra_workflows=[...])``.

The workflow file imports the program reference with a **plain import** (no
``imports_passed_through()`` dance) -- the decoupling that makes "bring your own
workflow" first-class -- and this test exercises exactly that path through the
sandbox.
"""

import importlib
import sys
import uuid
from pathlib import Path

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.converter import data_converter

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def compose_example():
    """Import the compose example and bind its program.

    ``compose_refs`` declares "compose_qa" with a side-effect-free
    ``dt.program(...)``; binding the implementation is the worker's job. We do that
    here (inside the test's registry-snapshot window, which the autouse
    ``restore_registry`` fixture rolls back afterward) so the program activity can
    build it. ``compose_program`` imports the ref with a plain import -- no
    passthrough block -- so this also proves that import path is sandbox-safe.
    """
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        compose_program = importlib.import_module("compose_program")
        compose_program.triage_agent.bind(
            lambda: dspy.ChainOfThought("question -> answer")
        )
        yield compose_program
    finally:
        sys.path.remove(str(EXAMPLES_DIR))


@pytest.mark.asyncio
async def test_compose_agent_run_inside_user_workflow(compose_example, dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(
            env.client,
            task_queue=task_queue,
            extra_workflows=[compose_example.ResearchWorkflow],
        )
        async with worker:
            answer = await env.client.execute_workflow(
                compose_example.ResearchWorkflow.run,
                "What color is the sky?",
                id=f"research-{uuid.uuid4().hex[:12]}",
                task_queue=task_queue,
            )

    # Both composed agent.run() calls dispatched the coarse activity and the
    # DummyLM's answer flowed back through the user workflow, shaped by the ref's
    # ``result`` adapter into a typed ``Answer``.
    assert answer.text == "blue"
