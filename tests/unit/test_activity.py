"""Activity-level tests using Temporal's ActivityEnvironment (no server)."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from temporalio.testing import ActivityEnvironment

import dspy_temporal as dt
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
