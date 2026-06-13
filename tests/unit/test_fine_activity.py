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
from dspy_temporal.fine.activities import lm_call_activity, tool_call_activity
from dspy_temporal.models import (
    LMCallInput,
    LMCallOutput,
    ToolCallInput,
    ToolCallOutput,
)


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
    assert output.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert output.model == "dummy"
    assert output.response_model == "dummy"


def test_lm_call_activity_without_worker_lm_raises():
    dt.clear_worker_lm()
    env = ActivityEnvironment()
    with pytest.raises(RuntimeError, match="requires a worker LM"):
        env.run(lm_call_activity, LMCallInput(messages=[{"role": "user", "content": "x"}]))


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
        env.run(lm_call_activity, LMCallInput(messages=[{"role": "user", "content": "x"}]))

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
    name = _register_weather_agent(lambda city: city)  # noqa: E731
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
