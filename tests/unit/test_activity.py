"""Activity-level tests using Temporal's ActivityEnvironment (no server)."""

import dataclasses
from datetime import timedelta

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from temporalio.testing import ActivityEnvironment

import dspy_temporal as dt
from dspy_temporal import config as config_mod
from dspy_temporal.coarse.activities import run_program_activity
from dspy_temporal.models import ProgramCallInput, ProgramCallOutput


def test_run_program_activity_returns_prediction(qa_program):
    env = ActivityEnvironment()
    call = ProgramCallInput(program="qa", inputs={"question": "color of the sky?"})

    # The activity is synchronous, so ActivityEnvironment.run returns directly.
    output = env.run(run_program_activity, call)

    assert isinstance(output, ProgramCallOutput)
    assert output.prediction["answer"] == "blue"
    assert "reasoning" in output.prediction


def test_runs_normally_with_heartbeat_timeout_set(qa_program):
    """The original bug: setting heartbeat_timeout made every coarse run fail.

    With the watchdog wrapping the program call, a configured heartbeat_timeout no
    longer self-destructs -- the activity heartbeats and completes normally. (A
    DummyLM run can finish before the first beat, so the 'beats fire' assertion
    lives in test_heartbeat.py with a controlled blocking body.)
    """
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_timeout=timedelta(seconds=2))
    call = ProgramCallInput(program="qa", inputs={"question": "color of the sky?"})

    output = env.run(run_program_activity, call)

    assert isinstance(output, ProgramCallOutput)
    assert output.prediction["answer"] == "blue"


def test_unknown_program_raises(qa_program):
    env = ActivityEnvironment()
    call = ProgramCallInput(program="does-not-exist", inputs={})
    with pytest.raises(KeyError):
        env.run(run_program_activity, call)


def test_predictor_own_lm_wins_over_worker_lm():
    """A predictor's bound .lm takes precedence over the worker default LM."""

    def build():
        m = dspy.ChainOfThought("question -> answer")
        m.set_lm(DummyLM([{"reasoning": "r", "answer": "red"}] * 5))
        return m

    dt.register_program("ownlm", build)
    dt.set_worker_lm(DummyLM([{"reasoning": "r", "answer": "blue"}] * 5))

    output = ActivityEnvironment().run(
        run_program_activity,
        ProgramCallInput(program="ownlm", inputs={"question": "?"}),
    )
    assert output.prediction["answer"] == "red"


def test_tracing_callback_not_double_added(qa_program):
    """If the same callback is also registered on dspy.settings, the activity
    applies it once, not twice (a double-add would double-emit every span)."""
    seen = {}

    class RecordingCallback:
        def on_module_start(self, call_id, instance, inputs):
            seen["callbacks"] = list(dspy.settings.callbacks or [])

    cb = RecordingCallback()
    dt.set_worker_lm(DummyLM([{"reasoning": "r", "answer": "blue"}] * 5))
    config_mod.set_tracing_callback(cb)

    # Simulate the callback already being present in dspy.settings.callbacks so
    # the activity must dedupe rather than append a second copy.
    with dspy.context(callbacks=[cb]):
        ActivityEnvironment().run(
            run_program_activity,
            ProgramCallInput(program="qa", inputs={"question": "?"}),
        )

    assert seen["callbacks"].count(cb) == 1


class _StaticProgram(dspy.Module):
    """Returns a Prediction without making any LM call."""

    def forward(self, **kwargs):
        return dspy.Prediction(answer="static")


def test_no_worker_lm_and_no_usage():
    """Covers the no-worker-LM branch and the lm_usage -> None path."""
    dt.clear_worker_lm()
    dt.register_program("static", _StaticProgram)

    output = ActivityEnvironment().run(
        run_program_activity,
        ProgramCallInput(program="static", inputs={}),
    )
    assert output.prediction["answer"] == "static"
    assert output.lm_usage is None
