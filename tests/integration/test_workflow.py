"""End-to-end workflow tests via a time-skipping Temporal test server."""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.config import CallOptions, RunConfig
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_program_runs_end_to_end(qa_program):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa",
                {"question": "color of the sky?"},
                task_queue=task_queue,
            )
    assert pred.answer == "blue"
    assert pred.reasoning


# A program that fails a configurable number of times before succeeding, to
# prove the workflow retries the activity per the retry policy.
_ATTEMPTS = {"n": 0}


class FlakyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, **kwargs):
        _ATTEMPTS["n"] += 1
        if _ATTEMPTS["n"] < 3:
            raise RuntimeError(f"transient failure #{_ATTEMPTS['n']}")
        return self.predict(**kwargs)


@pytest.mark.asyncio
async def test_activity_is_retried(dummy_lm):
    _ATTEMPTS["n"] = 0
    dt.register_program("flaky", FlakyProgram)
    dt.set_worker_lm(dummy_lm)

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    options = CallOptions(maximum_attempts=5, initial_interval_seconds=0.1)
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                "flaky",
                {"question": "color of the sky?"},
                task_queue=task_queue,
                options=options,
            )
    assert pred.answer == "blue"
    assert _ATTEMPTS["n"] == 3  # failed twice, succeeded on the third attempt
