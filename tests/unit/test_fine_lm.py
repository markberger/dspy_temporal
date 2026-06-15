"""Unit tests for the WorkflowLM seam (workflow-side, no Temporal server).

WorkflowLM.acall normally runs inside a workflow and turns each LM call into a
``dspy_lm_call`` activity. Here we monkeypatch the activity dispatch so we can
exercise the things acall does with the result without standing up a server:
return the already-processed outputs, replicate dspy.LM.forward's usage-tracker
side effect, and carry the right lm_ref / program / encoded kwargs in the input.
"""

import dspy
import pytest
from dspy.utils.usage_tracker import track_usage
from temporalio import workflow as tworkflow

from dspy_temporal.fine.lm import LM_ACTIVITY_NAME, WorkflowLM
from dspy_temporal.models import LMCallInput, LMCallOutput, LMSpec


def _spec(model="openai/x", **kw):
    return LMSpec(model=model, **kw)


@pytest.fixture
def fake_lm_activity(monkeypatch):
    """Replace workflow.execute_activity with a stub returning a canned result.

    Returns a setter so each test installs the LMCallOutput it wants and reads
    back the LMCallInput the seam built.
    """
    state = {}

    def install(output: LMCallOutput):
        state["output"] = output

        async def fake_execute_activity(name, arg, **kwargs):
            state["name"] = name
            state["arg"] = arg
            state["kwargs"] = kwargs
            return output

        monkeypatch.setattr(tworkflow, "execute_activity", fake_execute_activity)
        return state

    return install


def test_spec_drives_model_and_capability_flags():
    """The spec stands in for the real worker LM: model, kwargs, and the
    capability flags JSONAdapter branches on in the workflow."""
    lm = WorkflowLM(
        spec=_spec(
            model="openai/gpt-4o",
            supported_params=["response_format", "temperature"],
            supports_response_schema=True,
            supports_function_calling=True,
            kwargs={"temperature": 0.7, "max_tokens": 100},
        )
    )

    assert lm.model == "openai/gpt-4o"
    assert lm.kwargs == {"temperature": 0.7, "max_tokens": 100}
    assert lm.supported_params == {"response_format", "temperature"}
    assert lm.supports_response_schema is True
    assert lm.supports_function_calling is True


@pytest.mark.asyncio
async def test_acall_returns_outputs_and_feeds_usage_tracker(fake_lm_activity):
    state = fake_lm_activity(
        LMCallOutput(
            outputs=["the answer is blue"], usage={"total_tokens": 5}, model="openai/x"
        )
    )
    lm = WorkflowLM(spec=_spec(), lm_ref="predict", program="qa")

    # A usage tracker active (as Module.acall sets up via track_usage=True).
    with track_usage() as tracker:
        out = await lm.acall(
            messages=[{"role": "user", "content": "sky?"}], temperature=0.0
        )

    # Outputs pass straight through for the adapter to parse.
    assert out == ["the answer is blue"]
    # Dispatched to the right activity with a JSON-native LMCallInput carrying
    # the predictor's lm_ref + program and encoded kwargs.
    assert state["name"] == LM_ACTIVITY_NAME
    arg = state["arg"]
    assert isinstance(arg, LMCallInput)
    assert arg.messages == [{"role": "user", "content": "sky?"}]
    assert arg.lm_ref == "predict"
    assert arg.program == "qa"
    assert arg.lm_kwargs == {"temperature": 0.0}
    # Usage was attributed under the worker's request model (not the placeholder).
    assert tracker.usage_data["openai/x"]


@pytest.mark.asyncio
async def test_acall_routes_to_task_queue_when_set(fake_lm_activity):
    """A task_queue on the seam routes the dspy_lm_call activity to a dedicated
    pool; without one the key is absent (Temporal rejects task_queue=None)."""
    state = fake_lm_activity(LMCallOutput(outputs=["x"], usage={}))

    await WorkflowLM(spec=_spec(), task_queue="gpu-pool").acall(prompt="hi")
    assert state["kwargs"]["task_queue"] == "gpu-pool"

    await WorkflowLM(spec=_spec()).acall(prompt="hi")
    assert "task_queue" not in state["kwargs"]


@pytest.mark.asyncio
async def test_acall_without_tracker_or_usage_is_a_noop(fake_lm_activity):
    """The guard's false branch: no usage tracker -> nothing to feed, no error."""
    fake_lm_activity(LMCallOutput(outputs=["hi"], usage={}))
    lm = WorkflowLM(spec=_spec())

    # No track_usage() context -> dspy.settings.usage_tracker is None.
    assert dspy.settings.usage_tracker is None
    out = await lm.acall(prompt="hello")
    assert out == ["hi"]


def test_forward_is_guarded_against_sync_use():
    """The sync path would block the workflow thread; it must fail fast."""
    with pytest.raises(RuntimeError, match="only supports the async path"):
        WorkflowLM(spec=_spec()).forward(prompt="x")
