"""Fine-mode activity tests via Temporal's ActivityEnvironment (no server).

These exercise the two fine-mode activities in isolation: the LM call returns
already-processed outputs + usage read from an *isolated* history, and the tool
call runs the named tool (sync or async) and returns its observation.
"""

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from temporalio.testing import ActivityEnvironment

import dspy_temporal as dt
from dspy_temporal import config as config_mod
from dspy_temporal.fine.activities import (
    describe_lms_activity,
    lm_call_activity,
    tool_call_activity,
)
from dspy_temporal.models import (
    LMCallInput,
    LMCallOutput,
    LMDescribeInput,
    LMSpecsOutput,
    ToolCallInput,
    ToolCallOutput,
)


class _TwoPredictor(dspy.Module):
    """A module with two predictors; ``smart`` binds its own LM."""

    def __init__(self, bound_lm=None):
        super().__init__()
        self.fast = dspy.Predict("question -> answer")
        self.smart = dspy.Predict("question -> answer")
        if bound_lm is not None:
            self.smart.lm = bound_lm

    def forward(self, question):  # pragma: no cover - not executed here
        return self.fast(question=question)


# --- lm_call_activity --------------------------------------------------------


def test_lm_call_activity_returns_outputs_and_usage(dummy_lm):
    dt.set_worker_lm(dummy_lm)
    env = ActivityEnvironment()
    call = LMCallInput(messages=[{"role": "user", "content": "color of the sky?"}])

    output = env.run(lm_call_activity, call)

    assert isinstance(output, LMCallOutput)
    # ChatAdapter-formatted completion carrying the canned answer.
    assert "blue" in output.outputs[0]
    # Usage came from the isolated history[-1] entry; model is the worker LM's.
    assert output.usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert output.model == "dummy"
    assert output.response_model == "dummy"


def test_lm_call_activity_without_worker_lm_raises():
    dt.clear_worker_lm()
    env = ActivityEnvironment()
    with pytest.raises(RuntimeError, match="requires a worker LM"):
        env.run(
            lm_call_activity, LMCallInput(messages=[{"role": "user", "content": "x"}])
        )


def test_lm_call_activity_routes_by_lm_ref():
    """A predictor's bound .lm is honored via lm_ref; unbound predictors and a
    missing lm_ref fall back to the worker default."""
    dt.register_program(
        "multi", lambda: _TwoPredictor(bound_lm=DummyLM([{"answer": "BOUND"}] * 5))
    )
    dt.set_worker_lm(DummyLM([{"answer": "DEFAULT"}] * 5))
    env = ActivityEnvironment()
    msgs = [{"role": "user", "content": "q?"}]

    bound = env.run(
        lm_call_activity, LMCallInput(program="multi", lm_ref="smart", messages=msgs)
    )
    assert "BOUND" in bound.outputs[0]

    unbound = env.run(
        lm_call_activity, LMCallInput(program="multi", lm_ref="fast", messages=msgs)
    )
    assert "DEFAULT" in unbound.outputs[0]

    # No lm_ref (back-compat) -> worker default, never touches the registry.
    default = env.run(lm_call_activity, LMCallInput(messages=msgs))
    assert "DEFAULT" in default.outputs[0]


def test_lm_call_activity_decodes_response_format_marker(dummy_lm):
    """An encoded structured response_format is decoded to the litellm json_schema
    dict and accepted by the LM (DummyLM ignores it, but must not crash)."""
    dt.set_worker_lm(dummy_lm)
    env = ActivityEnvironment()
    marker = {
        "__dspy_temporal_response_format__": {
            "name": "Out",
            "json_schema": {"type": "object", "properties": {}},
        }
    }
    call = LMCallInput(
        messages=[{"role": "user", "content": "x"}],
        lm_kwargs={"response_format": marker, "temperature": 0.0},
    )
    out = env.run(lm_call_activity, call)
    assert isinstance(out, LMCallOutput)
    assert "blue" in out.outputs[0]


# --- describe_lms_activity ---------------------------------------------------


def test_describe_lms_activity_describes_each_predictor(dummy_lm):
    """Each predictor's effective LM is described; a bound .lm wins over the
    worker default, and there is a __default__ entry for the worker LM."""
    dt.register_program(
        "multi", lambda: _TwoPredictor(bound_lm=dspy.LM("openai/gpt-4o"))
    )
    dt.set_worker_lm(dummy_lm)
    env = ActivityEnvironment()

    out = env.run(describe_lms_activity, LMDescribeInput(program="multi"))

    assert isinstance(out, LMSpecsOutput)
    assert set(out.specs) == {"__default__", "fast", "smart"}
    # Worker default + the unbound predictor describe the dummy LM.
    assert out.specs["__default__"].model == "dummy"
    assert out.specs["fast"].model == "dummy"
    # The bound predictor describes gpt-4o, including its structured-output cap.
    assert out.specs["smart"].model == "openai/gpt-4o"
    assert out.specs["smart"].supports_response_schema is True
    assert "response_format" in out.specs["smart"].supported_params


def test_describe_lms_activity_strips_credentials():
    """An api_key in the LM's kwargs never crosses into an LMSpec."""
    dt.register_program("qa", lambda: dspy.Predict("question -> answer"))
    dt.set_worker_lm(
        dspy.LM("openai/gpt-4o-mini", api_key="sk-secret", temperature=0.0)
    )
    env = ActivityEnvironment()

    out = env.run(describe_lms_activity, LMDescribeInput(program="qa"))

    for spec in out.specs.values():
        assert "api_key" not in spec.kwargs
    assert out.specs["__default__"].kwargs.get("temperature") == 0.0


def test_describe_lms_activity_without_worker_lm_raises():
    dt.clear_worker_lm()
    dt.register_program("qa", lambda: dspy.Predict("question -> answer"))
    env = ActivityEnvironment()
    with pytest.raises(RuntimeError, match="requires a worker LM"):
        env.run(describe_lms_activity, LMDescribeInput(program="qa"))


def test_lm_call_activity_applies_tracing_callback_once(dummy_lm):
    """The tracing callback is applied (so the LM span is emitted in the
    activity) and deduped against dspy.settings.callbacks."""
    seen = {}

    class RecordingCallback:
        def on_lm_start(self, call_id, instance, inputs):
            seen["callbacks"] = list(dspy.settings.callbacks or [])

        def on_lm_end(self, call_id, outputs, exception=None):
            pass

    cb = RecordingCallback()
    dt.set_worker_lm(dummy_lm)
    config_mod.set_tracing_callback(cb)

    env = ActivityEnvironment()
    # Simulate the same callback also registered globally -> must not double-add.
    with dspy.context(callbacks=[cb]):
        env.run(
            lm_call_activity, LMCallInput(messages=[{"role": "user", "content": "x"}])
        )

    assert seen["callbacks"].count(cb) == 1


# --- tool_call_activity ------------------------------------------------------


def _register_weather_agent(tool):
    dt.register_program("agent", lambda: dspy.ReAct("question -> answer", tools=[tool]))
    return "agent"


def test_tool_call_activity_runs_sync_tool():
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"It is sunny in {city}."

    name = _register_weather_agent(get_weather)
    env = ActivityEnvironment()
    call = ToolCallInput(program=name, tool_name="get_weather", args={"city": "Tokyo"})

    output = env.run(tool_call_activity, call)

    assert isinstance(output, ToolCallOutput)
    assert output.observation == "It is sunny in Tokyo."


def test_tool_call_activity_runs_async_tool():
    """async tool bodies run from the sync activity via
    allow_tool_async_sync_conversion."""

    async def fetch_score(team: str) -> dict:
        """Return a score for a team."""
        return {"team": team, "score": 42}

    name = _register_weather_agent(fetch_score)
    env = ActivityEnvironment()
    call = ToolCallInput(program=name, tool_name="fetch_score", args={"team": "Reds"})

    output = env.run(tool_call_activity, call)

    assert output.observation == {"team": "Reds", "score": 42}


def test_tool_call_activity_unknown_tool_raises():
    name = _register_weather_agent(lambda city: city)
    env = ActivityEnvironment()
    call = ToolCallInput(program=name, tool_name="does_not_exist", args={})
    with pytest.raises(KeyError, match="no tool named"):
        env.run(tool_call_activity, call)


def test_tool_call_activity_program_without_tools_raises(qa_program):
    """A program with no .tools dict (plain ChainOfThought) is rejected."""
    env = ActivityEnvironment()
    call = ToolCallInput(program=qa_program, tool_name="anything", args={})
    with pytest.raises(KeyError, match="no tool named"):
        env.run(tool_call_activity, call)
