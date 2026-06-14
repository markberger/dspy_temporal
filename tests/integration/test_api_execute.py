"""End-to-end run_program (standalone path) via a time-skipping Temporal server."""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.config import RunConfig
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_program_start_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_exec",
        config=RunConfig(task_queue=task_queue),
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa_exec",
                {"question": "color of the sky?"},
                task_queue=task_queue,
            )

    assert pred.answer == "blue"
