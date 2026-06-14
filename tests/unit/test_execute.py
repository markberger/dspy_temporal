"""Unit tests for the reusable execute_* coroutines (no Temporal server).

execute_coarse / execute_fine normally run inside a workflow and dispatch our
activities. Here we monkeypatch ``workflow.execute_activity`` (the same pattern as
``test_fine_lm.py``) so we can exercise their data shaping and return values
without standing up a server: the right ProgramCallInput is built and the canned
ProgramCallOutput becomes a Prediction (coarse), and the describe/LM activities
are walked to bind a WorkflowLM and return a populated Prediction (fine).
"""

import dspy
import pytest
from temporalio import workflow as tworkflow

from dspy_temporal.execute import (
    ACTIVITY_NAME,
    DEFAULT_LM_REF,
    DESCRIBE_ACTIVITY_NAME,
    execute_coarse,
    execute_fine,
)
from dspy_temporal.fine.lm import LM_ACTIVITY_NAME
from dspy_temporal.models import (
    LMCallOutput,
    LMSpec,
    LMSpecsOutput,
    ProgramCallInput,
    ProgramCallOutput,
)
from dspy_temporal.options import CallOptions
from dspy_temporal.registry import register_program


@pytest.fixture
def dispatch(monkeypatch):
    """Route ``workflow.execute_activity`` to canned responses keyed by name.

    Returns a state dict: tests set ``state['responses'][name]`` and read back
    ``state['calls']`` (a list of (name, arg, kwargs)).
    """
    state = {"responses": {}, "calls": []}

    async def fake_execute_activity(name, arg, **kwargs):
        state["calls"].append((name, arg, kwargs))
        return state["responses"][name]

    monkeypatch.setattr(tworkflow, "execute_activity", fake_execute_activity)
    return state


# --- execute_coarse ----------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_coarse_builds_input_and_returns_prediction(dispatch):
    dispatch["responses"][ACTIVITY_NAME] = ProgramCallOutput(
        prediction={"answer": "blue"}, lm_usage={"dummy": {"total_tokens": 3}}
    )

    # options=None branch.
    pred = await execute_coarse("qa", {"question": "sky?"})

    assert pred.answer == "blue"
    # lm_usage was restored onto the prediction.
    assert pred.get_lm_usage() == {"dummy": {"total_tokens": 3}}
    name, arg, kwargs = dispatch["calls"][0]
    assert name == ACTIVITY_NAME
    assert isinstance(arg, ProgramCallInput)
    assert arg.program == "qa"
    assert arg.inputs == {"question": "sky?"}
    # options are kept OUT of the payload (the activity ignores them); they drive
    # the activity timeouts/retry instead. Default CallOptions applied here.
    assert arg.options is None
    assert kwargs["start_to_close_timeout"] == CallOptions().start_to_close_timeout()
    assert kwargs["retry_policy"] is not None


@pytest.mark.asyncio
async def test_execute_coarse_honors_supplied_options(dispatch):
    dispatch["responses"][ACTIVITY_NAME] = ProgramCallOutput(prediction={"answer": "x"})
    opts = CallOptions(maximum_attempts=7, start_to_close_timeout_seconds=12.0)

    # options-supplied branch.
    await execute_coarse("qa", {"question": "sky?"}, opts)

    _name, _arg, kwargs = dispatch["calls"][0]
    assert kwargs["start_to_close_timeout"] == opts.start_to_close_timeout()
    assert kwargs["retry_policy"].maximum_attempts == 7


# --- execute_fine ------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_fine_walks_predictors_and_returns_prediction(dispatch):
    """Drive a single-predictor program through fine orchestration with the
    describe + LM activities faked: the predictor's WorkflowLM dispatches one
    ``dspy_lm_call`` and the parsed field comes back on the prediction."""
    register_program("qa_fine_unit", lambda: dspy.Predict("question -> answer"))

    spec = LMSpec(model="dummy")
    dispatch["responses"][DESCRIBE_ACTIVITY_NAME] = LMSpecsOutput(
        specs={DEFAULT_LM_REF: spec}
    )
    # ChatAdapter parses the field-marked output into the `answer` field.
    dispatch["responses"][LM_ACTIVITY_NAME] = LMCallOutput(
        outputs=["[[ ## answer ## ]]\nblue\n\n[[ ## completed ## ]]"],
        usage={"total_tokens": 3},
        model="dummy",
    )

    pred = await execute_fine("qa_fine_unit", {"question": "color of the sky?"})

    assert pred.answer == "blue"
    # The describe activity ran first, then the LM call for the predictor.
    dispatched = [name for name, _arg, _kw in dispatch["calls"]]
    assert dispatched == [DESCRIBE_ACTIVITY_NAME, LM_ACTIVITY_NAME]
    # Usage tracking survived (WorkflowLM fed the tracker from the activity).
    assert pred.get_lm_usage()
