"""Fine-mode end-to-end tests via a time-skipping Temporal test server.

Proves the workflow-orchestrates / activities-do-the-I/O design: each LM call and
tool call is its own activity, usage tracking survives the boundary, the tool's
observation flows back into the answer, and completed steps are not re-run when a
later activity is retried (the core durability win over coarse mode).
"""

import uuid

import dspy
import pytest
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.config import CallOptions, RunConfig
from dspy_temporal.converter import data_converter


@pytest.mark.asyncio
async def test_fine_chain_of_thought_end_to_end(dummy_lm):
    """A ChainOfThought run in fine mode: one LM call -> one activity.

    Asserts the parsed fields come back AND that lm_usage is populated -- which
    only happens if WorkflowLM fed the usage tracker from the activity result.
    """
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    handle = dt.deploy_module(
        "qa_fine",
        lambda: dspy.ChainOfThought("question -> answer"),
        config=RunConfig(task_queue=task_queue, mode=dt.RunMode.FINE),
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await handle.execute(env.client, {"question": "color of the sky?"})

    assert pred.answer == "blue"
    assert pred.reasoning
    # Usage crossed the activity boundary and landed on the prediction.
    assert pred.get_lm_usage()
    assert "dummy" in pred.get_lm_usage()


@pytest.mark.asyncio
async def test_fine_react_tool_observation_influences_answer(fine_react):
    """A ReAct run in fine mode: the tool call is a separate activity whose
    observation flows into the final answer."""
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                fine_react.name,
                {"question": "What's the weather in Tokyo?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert "sunny" in pred.answer.lower()
    # The tool ran exactly once, as its own activity (not inlined, not re-run).
    assert fine_react.counters["tool"] == 1
    assert fine_react.counters["react"] == 2  # one tool-pick step, one finish step
    assert fine_react.counters["extract"] == 1


@pytest.mark.asyncio
async def test_fine_completed_steps_not_reexecuted_on_retry(fine_react):
    """The durability guarantee: when a *later* activity (the extract LM call)
    fails and is retried, the already-completed LM/tool activities are not
    re-run -- unlike coarse mode, which would replay the whole program."""
    # Worker LM that fails the first extract-step call, then succeeds.
    dt.set_worker_lm(fine_react.worker_lm_cls(fail_extract_once=True))

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    options = CallOptions(maximum_attempts=5, initial_interval_seconds=0.1)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                fine_react.name,
                {"question": "What's the weather in Tokyo?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
                options=options,
            )

    assert "sunny" in pred.answer.lower()
    # The extract activity was retried (failed once, then succeeded)...
    assert fine_react.counters["extract"] == 2
    # ...but the earlier, already-completed steps each ran exactly once.
    assert fine_react.counters["react"] == 2
    assert fine_react.counters["tool"] == 1
