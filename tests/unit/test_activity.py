"""Activity-level tests using Temporal's ActivityEnvironment (no server)."""

import pytest
from temporalio.testing import ActivityEnvironment

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
