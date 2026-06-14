"""End-to-end: a user @workflow.defn composes a deployed program.

Proves ``await agent.run(**inputs)`` works *inside* a user-authored workflow:
``TemporalProgram.run`` sees it is in a workflow and dispatches ``execute_coarse``
(our program activity) inline. The user workflow (``ResearchWorkflow`` from
examples/compose_program.py) chains two such calls across a ``workflow.sleep`` and
is served on the same worker via ``build_worker(extra_workflows=[...])``.
"""

import importlib
import sys
import uuid
from pathlib import Path

import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.converter import data_converter

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def compose_example():
    """Import the compose example (registers "compose_qa" + ResearchWorkflow).

    Forces a fresh execution of the module body (evicting it -- and the
    ``compose_agents`` module it passthrough-imports its ``deploy`` from -- from
    ``sys.modules`` first) so the import-time ``deploy`` re-runs inside this
    test's registry-snapshot window -- the autouse ``restore_registry`` fixture
    rolls back each test's registrations, so a once-per-session import would
    leave "compose_qa" unregistered by the time this test runs.
    """
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        sys.modules.pop("compose_program", None)
        sys.modules.pop("compose_agents", None)
        compose_program = importlib.import_module("compose_program")
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
    # DummyLM's answer flowed back through the user workflow.
    assert answer == "blue"
