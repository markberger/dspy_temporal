"""End-to-end: run a bound program both ways via a time-skipping server.

``ref.start`` is the reference path -- the ref's own mode plus an explicit serving
queue. ``run_program`` is kept as the low-level by-name escape hatch. The
non-blocking siblings (``ref.start_nowait`` + ``ref.result_of`` and the by-name
``start_program_nowait`` + ``prediction_of``) hand back a handle and decode it
later -- including across a fresh ``get_workflow_handle(id)``.
"""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.client import prediction_of
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_ref_start_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    agent = dt.program("qa_exec").bind(
        lambda: dspy.ChainOfThought("question -> answer")
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            # The ref carries its mode; the serving queue is passed explicitly.
            pred = await agent.start(
                env.client, task_queue=task_queue, question="color of the sky?"
            )

    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_run_program_by_name_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.program("qa_exec_byname").bind(lambda: dspy.ChainOfThought("question -> answer"))
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


@pytest.mark.asyncio
async def test_ref_start_nowait_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    agent = dt.program("qa_nowait").bind(
        lambda: dspy.ChainOfThought("question -> answer")
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            # Start now: a handle comes back immediately (the result is not awaited).
            handle = await agent.start_nowait(
                env.client, task_queue=task_queue, question="color of the sky?"
            )
            # Poll later: re-obtain the handle by id (the cross-request shape, with
            # no result type pinned) and decode it through the ref.
            polled = env.client.get_workflow_handle(handle.id)
            pred = await agent.result_of(polled)

    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_start_program_nowait_by_name_end_to_end(dummy_lm):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.program("qa_nowait_byname").bind(
        lambda: dspy.ChainOfThought("question -> answer")
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            handle = await dt.start_program_nowait(
                env.client,
                "qa_nowait_byname",
                {"question": "color of the sky?"},
                task_queue=task_queue,
            )
            pred = await prediction_of(handle)

    assert pred.answer == "blue"
