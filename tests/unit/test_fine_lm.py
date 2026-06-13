"""Unit tests for the WorkflowLM seam (workflow-side, no Temporal server).

WorkflowLM.acall normally runs inside a workflow and turns each LM call into a
``dspy_lm_call`` activity. Here we monkeypatch the activity dispatch so we can
exercise the two things acall does with the result without standing up a server:
return the already-processed outputs, and (when a usage tracker is active)
replicate dspy.LM.forward's usage-tracker side effect.
"""

import dspy
import pytest
from dspy.utils.usage_tracker import track_usage
from temporalio import workflow as tworkflow

from dspy_temporal.fine.lm import LM_ACTIVITY_NAME, WorkflowLM
from dspy_temporal.models import LMCallInput, LMCallOutput


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


@pytest.mark.asyncio
async def test_acall_returns_outputs_and_feeds_usage_tracker(fake_lm_activity):
    state = fake_lm_activity(
        LMCallOutput(outputs=["the answer is blue"], usage={"total_tokens": 5}, model="openai/x")
    )
    lm = WorkflowLM()

    # A usage tracker active (as Module.acall sets up via track_usage=True).
    with track_usage() as tracker:
        out = await lm.acall(messages=[{"role": "user", "content": "sky?"}])

    # Outputs pass straight through for the adapter to parse.
    assert out == ["the answer is blue"]
    # Dispatched to the right activity with a JSON-native LMCallInput.
    assert state["name"] == LM_ACTIVITY_NAME
    assert isinstance(state["arg"], LMCallInput)
    assert state["arg"].messages == [{"role": "user", "content": "sky?"}]
    # Usage was attributed under the worker's request model (not the placeholder).
    assert tracker.usage_data["openai/x"]


@pytest.mark.asyncio
async def test_acall_without_tracker_or_usage_is_a_noop(fake_lm_activity):
    """The guard's false branch: no usage tracker -> nothing to feed, no error."""
    fake_lm_activity(LMCallOutput(outputs=["hi"], usage={}))
    lm = WorkflowLM()

    # No track_usage() context -> dspy.settings.usage_tracker is None.
    assert dspy.settings.usage_tracker is None
    out = await lm.acall(prompt="hello")
    assert out == ["hi"]


def test_forward_is_guarded_against_sync_use():
    """The sync path would block the workflow thread; it must fail fast."""
    with pytest.raises(RuntimeError, match="only supports the async path"):
        WorkflowLM().forward(prompt="x")
