"""End-to-end: a Worker wired with DSPyPlugin serves DSPy programs.

Proves the declarative path: ``Worker(client, task_queue=..., plugins=[DSPyPlugin()])``
contributes the activities, both generic workflows, and the DSPy sandbox runner,
so ``dt.run_program`` works against it exactly like a ``build_worker`` worker.
"""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

import dspy_temporal as dt
from dspy_temporal.config import RunConfig
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_end_to_end_worker_with_plugin(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_plugin",
        config=RunConfig(task_queue=task_queue),
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = Worker(
            env.client,
            task_queue=task_queue,
            plugins=[dt.DSPyPlugin()],
        )
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa_plugin",
                {"question": "color of the sky?"},
                task_queue=task_queue,
            )

    assert pred.answer == "blue"
    assert pred.reasoning


@pytest.mark.asyncio
async def test_plugin_on_client_installs_converter_and_propagates(dummy_lm):
    """A single DSPyPlugin() on the *client* installs the pydantic converter and,
    being a worker plugin too, propagates the DSPy set to a bare Worker (#11)."""
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_propagate",
        config=RunConfig(task_queue=task_queue),
    )
    dt.set_worker_lm(dummy_lm)

    # No explicit data_converter: the plugin's configure_client must install it.
    async with await WorkflowEnvironment.start_time_skipping(
        plugins=[dt.DSPyPlugin()]
    ) as env:
        assert env.client.data_converter is data_converter

        # Bare Worker (no plugins=): the combined plugin propagates from the client.
        worker = Worker(env.client, task_queue=task_queue)
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa_propagate",
                {"question": "color of the sky?"},
                task_queue=task_queue,
            )

    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_plugin_replays_recorded_history(dummy_lm):
    """Replayer(plugins=[DSPyPlugin()]) can replay a recorded DSPy history (#11):
    the plugin's configure_replayer supplies the workflows, sandbox runner, and
    pydantic converter the replayer needs to decode and re-execute the history."""
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    workflow_id = f"dspy-replay-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_replay",
        config=RunConfig(task_queue=task_queue),
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            await dt.run_program(
                env.client,
                "qa_replay",
                {"question": "color of the sky?"},
                task_queue=task_queue,
                workflow_id=workflow_id,
            )
        history = await env.client.get_workflow_handle(workflow_id).fetch_history()

    # workflows=[] is the empty seed the plugin extends with DSPY_WORKFLOWS, and
    # the plugin also supplies the passthrough sandbox runner the DSPy workflows
    # require -- without either, replay_workflow raises. (The pydantic converter
    # is also installed here, but isn't load-bearing for this history; its
    # replayer wiring is asserted directly in test_plugin.py.)
    replayer = Replayer(workflows=[], plugins=[dt.DSPyPlugin()])
    await replayer.replay_workflow(history)
