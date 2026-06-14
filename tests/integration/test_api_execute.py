"""End-to-end: start a deployed program both ways via a time-skipping server.

``handle.start`` is the primary path -- it proves the handle is the single source
of truth for queue + mode (the caller passes neither). ``run_program`` is kept as
the low-level by-name escape hatch.
"""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_handle_start_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    agent = dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_exec",
        task_queue=task_queue,
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            # The handle knows its own queue + mode -- the caller passes neither.
            pred = await agent.start(env.client, question="color of the sky?")

    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_run_program_by_name_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_exec_byname",
        task_queue=task_queue,
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa_exec_byname",
                {"question": "color of the sky?"},
                task_queue=task_queue,
            )

    assert pred.answer == "blue"
