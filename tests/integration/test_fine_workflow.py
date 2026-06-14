"""Fine-mode end-to-end tests via a time-skipping Temporal test server.

Proves the workflow-orchestrates / activities-do-the-I/O design: each LM call and
tool call is its own activity, usage tracking survives the boundary, the tool's
observation flows back into the answer, and completed steps are not re-run when a
later activity is retried (the core durability win over coarse mode).
"""

import uuid

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.config import CallOptions
from dspy_temporal.converter import data_converter


class _NamedDummyLM(DummyLM):
    """A DummyLM with a distinct model id, so per-predictor routing is visible
    in ``lm_usage`` (which is keyed by the resolved LM's model)."""

    def __init__(self, model, answers):
        super().__init__(answers)
        self.model = model


# Captures the response_format the activity-side LM actually receives, so the
# JSONAdapter test can prove the structured format crossed the boundary (DummyLM
# itself ignores response_format, so the answer alone wouldn't prove it).
_RF_SEEN = {}


class _StructuredDummyLM(DummyLM):
    """A DummyLM that *claims* structured-output support, so JSONAdapter takes
    the native ``response_format`` path (which DummyLM then ignores, returning its
    canned JSON answer). Records the response_format it receives so the test can
    assert the pydantic schema crossed the activity boundary."""

    @property
    def supported_params(self):
        return {"response_format", "temperature", "max_tokens"}

    @property
    def supports_response_schema(self):
        return True

    def forward(self, prompt=None, messages=None, **kwargs):
        if kwargs.get("response_format") is not None:
            _RF_SEEN["response_format"] = kwargs["response_format"]
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class _TwoStage(dspy.Module):
    """Two predictors where the second binds its own (distinct) LM."""

    def __init__(self):
        super().__init__()
        self.stage1 = dspy.Predict("question -> topic")
        self.stage2 = dspy.Predict("topic -> answer")
        self.stage2.lm = _NamedDummyLM("model-smart", [{"answer": "sunny"}] * 5)

    async def aforward(self, question):
        topic = await self.stage1.acall(question=question)
        return await self.stage2.acall(topic=topic.topic)


class _CompiledInner(dspy.Module):
    """A sub-module marked ``_compiled`` (as an optimizer would), whose predictor
    binds its own distinct LM. DSPy's ``named_predictors()`` skips compiled
    sub-modules, so without ``all_named_predictors`` this predictor would be
    invisible to fine mode -- never bound a ``WorkflowLM``, its real LM running
    inside the workflow sandbox instead of a recorded ``dspy_lm_call`` activity."""

    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
        self.qa.lm = _NamedDummyLM("model-compiled", [{"answer": "blue"}] * 5)
        self._compiled = True

    async def aforward(self, question):
        return await self.qa.acall(question=question)


class _OuterWithCompiled(dspy.Module):
    """Wraps a compiled sub-module as its only predictor-bearing child."""

    def __init__(self):
        super().__init__()
        self.inner = _CompiledInner()

    async def aforward(self, question):
        return await self.inner.acall(question=question)


@pytest.mark.asyncio
async def test_fine_chain_of_thought_end_to_end(dummy_lm):
    """A ChainOfThought run in fine mode: one LM call -> one activity.

    Asserts the parsed fields come back AND that lm_usage is populated -- which
    only happens if WorkflowLM fed the usage tracker from the activity result.
    """
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="qa_fine",
        task_queue=task_queue,
        mode=dt.RunMode.FINE,
    )
    dt.set_worker_lm(dummy_lm)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa_fine",
                {"question": "color of the sky?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert pred.answer == "blue"
    assert pred.reasoning
    # Usage crossed the activity boundary and landed on the prediction.
    assert pred.get_lm_usage()
    assert "dummy" in pred.get_lm_usage()


@pytest.mark.asyncio
async def test_fine_per_predictor_multi_lm(dummy_lm):
    """Each predictor's effective LM is honored: the unbound predictor uses the
    worker default, the bound one uses its own LM -- visible as two distinct
    model keys in lm_usage (two separate dspy_lm_call activities)."""
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        _TwoStage,
        name="two_stage",
        task_queue=task_queue,
        mode=dt.RunMode.FINE,
    )
    dt.set_worker_lm(_NamedDummyLM("model-fast", [{"topic": "weather"}] * 5))

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            pred = await dt.run_program(
                env.client,
                "two_stage",
                {"question": "weather in Tokyo?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert pred.answer == "sunny"
    usage = pred.get_lm_usage()
    # Both LMs were used, each in its own activity, attributed to its own model.
    assert "model-fast" in usage  # stage1 -> worker default
    assert "model-smart" in usage  # stage2 -> bound .lm


@pytest.mark.asyncio
async def test_fine_compiled_submodule_predictor_routes_through_activity():
    """A predictor inside a *compiled* sub-module is still discovered in fine
    mode: it gets a WorkflowLM bound to its own LM, so its call runs as a
    dspy_lm_call activity (its model id appears in lm_usage) rather than the
    real LM executing inside the workflow sandbox. Guards the all_named_predictors
    fix end-to-end -- with plain named_predictors() this predictor is invisible."""
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        _OuterWithCompiled,
        name="compiled_inner",
        task_queue=task_queue,
        mode=dt.RunMode.FINE,
    )
    # A distinct worker default so we can prove the *bound* (compiled) LM ran,
    # not the fallback.
    dt.set_worker_lm(_NamedDummyLM("model-default", [{"answer": "red"}] * 5))

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
        async with worker:
            pred = await dt.run_program(
                env.client,
                "compiled_inner",
                {"question": "color of the sky?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert pred.answer == "blue"  # the compiled predictor's bound LM answered
    usage = pred.get_lm_usage()
    # The compiled predictor was routed to its own LM via its own activity...
    assert "model-compiled" in usage
    # ...and the worker default was never needed (no unbound predictor).
    assert "model-default" not in usage


@pytest.mark.asyncio
async def test_fine_json_adapter_structured_output():
    """A program run under JSONAdapter in fine mode: the structured
    response_format (a pydantic class) crosses the boundary and the JSON answer
    parses back -- the ChatAdapter-only limitation is lifted."""
    _RF_SEEN.clear()
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    dt.deploy(
        lambda: dspy.Predict("question -> answer"),
        name="qa_json",
        task_queue=task_queue,
        mode=dt.RunMode.FINE,
    )
    # Worker LM claims response_format support and emits JSON-formatted answers.
    dt.set_worker_lm(
        _StructuredDummyLM([{"answer": "blue"}] * 5, adapter=dspy.JSONAdapter())
    )

    saved_adapter = dspy.settings.adapter
    dspy.configure(adapter=dspy.JSONAdapter())
    try:
        async with await WorkflowEnvironment.start_time_skipping(
            data_converter=data_converter
        ) as env:
            worker = dt.build_worker(env.client, task_queue=task_queue)
            async with worker:
                pred = await dt.run_program(
                    env.client,
                    "qa_json",
                    {"question": "color of the sky?"},
                    task_queue=task_queue,
                    mode=dt.RunMode.FINE,
                )
    finally:
        dspy.configure(adapter=saved_adapter)

    assert pred.answer == "blue"
    # The structured response_format crossed the boundary and was decoded into
    # the litellm json_schema form (it would be absent if it had been dropped).
    rf = _RF_SEEN.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert "answer" in rf["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_fine_react_tool_observation_influences_answer(fine_react):
    """A ReAct run in fine mode: the tool call is a separate activity whose
    observation flows into the final answer."""
    task_queue = f"tq-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter
    ) as env:
        worker = dt.build_worker(env.client, task_queue=task_queue)
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
        worker = dt.build_worker(env.client, task_queue=task_queue)
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
