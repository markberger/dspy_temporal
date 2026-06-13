"""Win C end-to-end: a Worker wired with DSPyPlugin serves DSPy programs.

Proves the declarative path: ``Worker(client, task_queue=..., plugins=[DSPyPlugin()])``
contributes the activities, both generic workflows, and the DSPy sandbox runner,
so ``dt.run_program`` works against it exactly like a ``build_worker`` worker.
"""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import dspy_temporal as dt
from dspy_temporal.config import RunConfig
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_end_to_end_worker_with_plugin(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy_module(
        "qa_plugin",
        lambda: dspy.ChainOfThought("question -> answer"),
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
