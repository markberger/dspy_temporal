"""End-to-end test for the two-LM fine-mode example (examples/two_lm_program.py).

Proves the per-predictor multi-LM path on the *actual* example: each predictor
is bound to a distinct model, and a single fine-mode run routes each predictor's
call to its own LM -- visible as two distinct model keys in lm_usage (i.e. two
separate ``dspy_lm_call`` activities). Uses named DummyLMs carrying the example's
real model ids, so it runs offline / in CI with no network or API key.
"""

import sys
import uuid
from pathlib import Path

import pytest
from dspy.utils.dummies import DummyLM
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.converter import data_converter

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def two_lm_example():
    """Import the example module (registers 'two_lm_qa') and expose its constants."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        import two_lm_program
    finally:
        sys.path.remove(str(EXAMPLES_DIR))
    return two_lm_program


class _NamedDummyLM(DummyLM):
    """A DummyLM with a distinct model id, so per-predictor routing is visible in
    lm_usage (which is keyed by the resolved LM's model)."""

    def __init__(self, model, answers):
        super().__init__(answers)
        self.model = model


def test_example_binds_two_distinct_models(two_lm_example):
    """The example wires a *different* LM onto each predictor."""
    program = two_lm_example.TwoLMQA()
    models = {
        p.lm.model for _, p in program.named_predictors() if getattr(p, "lm", None)
    }
    assert models == {two_lm_example.DRAFT_MODEL, two_lm_example.REFINE_MODEL}
    assert two_lm_example.DRAFT_MODEL != two_lm_example.REFINE_MODEL


@pytest.mark.asyncio
async def test_two_lm_example_routes_each_predictor(two_lm_example):
    """A fine-mode run uses both bound models -- one dspy_lm_call activity each --
    so lm_usage carries both model ids."""
    draft_model = two_lm_example.DRAFT_MODEL
    refine_model = two_lm_example.REFINE_MODEL

    def build_offline_two_lm():
        # Reuse the real example module, then swap its OpenRouter LMs for named
        # dummies so the test is offline + deterministic (the structure -- two
        # predictors, each with its own LM -- is the example's).
        program = two_lm_example.TwoLMQA()
        program.draft.set_lm(
            _NamedDummyLM(
                draft_model, [{"reasoning": "r", "draft_answer": "the sky is blue"}] * 5
            )
        )
        program.refine.set_lm(
            _NamedDummyLM(refine_model, [{"reasoning": "r", "answer": "blue"}] * 5)
        )
        return program

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.program("two_lm_qa_offline", mode=dt.RunMode.FINE).bind(build_offline_two_lm)
    # The worker default backs the __default__ spec/fallback; named so it can't be
    # confused with the two per-predictor models.
    dt.set_worker_lm(_NamedDummyLM("default-model", [{"answer": "blue"}] * 5))

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            pred = await dt.run_program(
                env.client,
                "two_lm_qa_offline",
                {"question": "Why is the sky blue?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert pred.answer == "blue"
    usage = pred.get_lm_usage()
    # Each predictor was attributed to its own model -> two distinct keys.
    assert draft_model in usage  # draft predictor -> DRAFT_MODEL
    assert refine_model in usage  # refine predictor -> REFINE_MODEL
