"""End-to-end workflow tests via a time-skipping Temporal test server."""

import time
import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.config import CallOptions, clear_worker_lm
from dspy_temporal.converter import data_converter
from dspy_temporal.registry import register_program


@pytest.mark.asyncio
async def test_program_runs_end_to_end(qa_program):
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
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
    register_program("flaky", FlakyProgram)
    dt.set_worker_lm(dummy_lm)

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    options = CallOptions(maximum_attempts=5, initial_interval_seconds=0.1)
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
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


# A coarse program that blocks (real wall-clock) for longer than its
# heartbeat_timeout. It makes no LM call -- the point is to occupy the activity
# past the timeout so that completing at all proves the watchdog kept beating.
_HEARTBEAT_TIMEOUT_SECONDS = 1.0
_BLOCK_SECONDS = 2.5


class SlowProgram(dspy.Module):
    def forward(self, **kwargs):
        time.sleep(_BLOCK_SECONDS)
        return dspy.Prediction(answer="slept")


@pytest.mark.asyncio
async def test_coarse_activity_heartbeats_past_its_timeout():
    """Exercises the *real* worker heartbeat path (not ActivityEnvironment's stub).

    The activity blocks ~2.5s with a 1.0s heartbeat_timeout and a single attempt,
    so the server would HEARTBEAT-timeout and fail it unless the watchdog actually
    delivers beats. A green run proves heartbeats reach the server -- and guards
    the #8/#10 interaction: if the coarse activity is made ``async def`` again, the
    watchdog can't beat from its daemon thread (asyncio.create_task has no loop in
    that thread) and this test times out.

    Auto time-skipping is disabled so the heartbeat_timeout is enforced against
    real wall-clock; otherwise the test server could skip the timeout timer
    forward before any real beat lands.
    """
    clear_worker_lm()  # SlowProgram makes no LM call
    register_program("slow", SlowProgram)

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    options = CallOptions(
        heartbeat_timeout_seconds=_HEARTBEAT_TIMEOUT_SECONDS,
        start_to_close_timeout_seconds=30.0,
        maximum_attempts=1,  # a missed heartbeat fails outright (no masking retry)
    )
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            # auto_time_skipping_disabled is a sync CM; wrap the awaited run so the
            # heartbeat_timeout is enforced in real wall-clock.
            with env.auto_time_skipping_disabled():
                pred = await dt.run_program(
                    env.client,
                    "slow",
                    {"question": "?"},
                    task_queue=task_queue,
                    options=options,
                )
    assert pred.answer == "slept"
