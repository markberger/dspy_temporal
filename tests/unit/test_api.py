"""Tests for the auto-wrap API: deploy_module + DeployedProgram.execute.

`execute` is covered against a fake client (records the call) so no Temporal
server is needed; an end-to-end `.execute()` lives in the integration suite.
"""

import re

import dspy
import pytest

import dspy_temporal as dt
from dspy_temporal.coarse.api import DeployedProgram
from dspy_temporal.config import CallOptions, RunConfig
from dspy_temporal.models import ProgramCallInput, ProgramCallOutput


class FakeClient:
    """Records the execute_workflow call and returns a canned output."""

    def __init__(self):
        self.calls = []

    async def execute_workflow(self, run, call, *, id, task_queue):
        self.calls.append(
            {"run": run, "call": call, "id": id, "task_queue": task_queue}
        )
        return ProgramCallOutput(prediction={"answer": "blue"})


def test_deploy_module_registers_and_returns_handle():
    handle = dt.deploy_module("qa2", lambda: dspy.Predict("q -> a"))
    assert isinstance(handle, DeployedProgram)
    assert handle.name == "qa2"
    assert "qa2" in dt.default_registry()
    # Default config when none supplied.
    assert handle.config.task_queue == "dspy-temporal"


def test_deploy_module_uses_given_config():
    cfg = RunConfig(task_queue="tq-custom")
    handle = dt.deploy_module("qa3", lambda: dspy.Predict("q -> a"), config=cfg)
    assert handle.config is cfg


@pytest.mark.asyncio
async def test_execute_generates_default_workflow_id():
    handle = DeployedProgram(name="qa", config=RunConfig(task_queue="tq"))
    client = FakeClient()

    pred = await handle.execute(client, {"question": "sky?"})

    assert pred.answer == "blue"
    call = client.calls[0]
    assert re.fullmatch(r"dspy-qa-[0-9a-f]{12}", call["id"])
    assert call["task_queue"] == "tq"
    assert isinstance(call["call"], ProgramCallInput)
    # options default to the config's call_options
    assert call["call"].options == RunConfig().call_options


@pytest.mark.asyncio
async def test_execute_honors_overrides():
    handle = DeployedProgram(name="qa", config=RunConfig(task_queue="tq"))
    client = FakeClient()
    opts = CallOptions(maximum_attempts=9)

    await handle.execute(
        client,
        {"question": "sky?"},
        workflow_id="wf-explicit",
        task_queue="tq-override",
        options=opts,
    )

    call = client.calls[0]
    assert call["id"] == "wf-explicit"
    assert call["task_queue"] == "tq-override"
    assert call["call"].options.maximum_attempts == 9
