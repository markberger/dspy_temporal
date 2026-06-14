"""Tests for the auto-wrap API: deploy + the TemporalProgram handle, and the
standalone run_program client helper.

`run_program` is covered against a fake client (records the call) so no Temporal
server is needed; an end-to-end `run_program` lives in the integration suite.
`.run`'s in-workflow dispatch is covered by monkeypatching the execute_*
coroutines; its outside-a-workflow degrade runs a real local DSPy call.
"""

import re

import dspy
import pytest

import dspy_temporal as dt
from dspy_temporal.coarse import api as api_mod
from dspy_temporal.coarse.api import TemporalProgram
from dspy_temporal.config import CallOptions, RunConfig, RunMode
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


# --- deploy ------------------------------------------------------------------


def test_deploy_builder_registers_and_returns_handle():
    handle = dt.deploy(lambda: dspy.Predict("q -> a"), name="qa2")
    assert isinstance(handle, TemporalProgram)
    assert handle.name == "qa2"
    assert "qa2" in dt.default_registry()
    # defaults: coarse + the default task queue.
    assert handle.config.mode == RunMode.COARSE
    assert handle.config.task_queue == "dspy-temporal"


def test_deploy_with_instance_registers_and_returns_handle():
    handle = dt.deploy(
        dspy.ChainOfThought("question -> answer"),
        name="inst",
        mode=RunMode.FINE,
        task_queue="tq-inst",
    )
    assert isinstance(handle, TemporalProgram)
    assert "inst" in dt.default_registry()
    # config assembled from mode + task_queue when none supplied.
    assert handle.config.mode == RunMode.FINE
    assert handle.config.task_queue == "tq-inst"
    # The registered prototype builds an LM-stripped copy.
    built = dt.default_registry().build("inst")
    assert all(p.lm is None for _n, p in built.named_predictors())


def test_deploy_supplied_config_takes_precedence():
    cfg = RunConfig(task_queue="tq-explicit", mode=RunMode.FINE)
    # mode/task_queue kwargs are ignored in favor of the supplied config.
    handle = dt.deploy(
        lambda: dspy.Predict("q -> a"),
        name="cfg",
        mode=RunMode.COARSE,
        task_queue="tq-ignored",
        config=cfg,
    )
    assert handle.config is cfg


# --- run_program (standalone path) -------------------------------------------


@pytest.mark.asyncio
async def test_run_program_generates_default_workflow_id():
    client = FakeClient()

    pred = await dt.run_program(client, "qa", {"question": "sky?"}, task_queue="tq")

    assert pred.answer == "blue"
    call = client.calls[0]
    assert re.fullmatch(r"dspy-qa-[0-9a-f]{12}", call["id"])
    assert call["task_queue"] == "tq"
    assert isinstance(call["call"], ProgramCallInput)


@pytest.mark.asyncio
async def test_run_program_honors_overrides():
    client = FakeClient()
    opts = CallOptions(maximum_attempts=9)

    await dt.run_program(
        client,
        "qa",
        {"question": "sky?"},
        task_queue="tq-override",
        workflow_id="wf-explicit",
        options=opts,
    )

    call = client.calls[0]
    assert call["id"] == "wf-explicit"
    assert call["task_queue"] == "tq-override"
    assert call["call"].options.maximum_attempts == 9


@pytest.mark.asyncio
async def test_run_program_selects_fine_workflow_for_fine_mode():
    from dspy_temporal.fine.workflow import DSPyProgramFineWorkflow

    client = FakeClient()
    await dt.run_program(client, "qa", {"question": "sky?"}, mode=RunMode.FINE)
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


# --- .run: context-aware dispatch -------------------------------------------


@pytest.mark.asyncio
async def test_run_in_workflow_coarse_dispatches_execute_coarse(monkeypatch):
    recorded = {}

    async def fake_execute_coarse(name, inputs, options):
        recorded["name"] = name
        recorded["inputs"] = inputs
        recorded["options"] = options
        return dspy.Prediction(answer="from_coarse")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_coarse", fake_execute_coarse)
    monkeypatch.setattr(api_mod, "execute_fine", _should_not_call)

    handle = TemporalProgram(name="qa", config=RunConfig(mode=RunMode.COARSE))
    pred = await handle.run(question="sky?")

    assert pred.answer == "from_coarse"
    assert recorded["name"] == "qa"
    assert recorded["inputs"] == {"question": "sky?"}
    # The context-aware path carries no per-handle options.
    assert recorded["options"] is None


@pytest.mark.asyncio
async def test_run_in_workflow_fine_dispatches_execute_fine(monkeypatch):
    recorded = {}

    async def fake_execute_fine(name, inputs, options):
        recorded["name"] = name
        return dspy.Prediction(answer="from_fine")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_fine", fake_execute_fine)
    monkeypatch.setattr(api_mod, "execute_coarse", _should_not_call)

    handle = TemporalProgram(name="qa", config=RunConfig(mode=RunMode.FINE))
    pred = await handle.run(question="sky?")

    assert pred.answer == "from_fine"
    assert recorded["name"] == "qa"


@pytest.mark.asyncio
async def test_run_outside_workflow_coarse_runs_in_process(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    dt.deploy(lambda: dspy.ChainOfThought("question -> answer"), name="degrade_coarse")

    handle = TemporalProgram(
        name="degrade_coarse", config=RunConfig(mode=RunMode.COARSE)
    )
    with dspy.context(lm=dummy_lm):
        pred = await handle.run(question="color of the sky?")
    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_run_outside_workflow_fine_runs_in_process(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    dt.deploy(lambda: dspy.ChainOfThought("question -> answer"), name="degrade_fine")

    handle = TemporalProgram(name="degrade_fine", config=RunConfig(mode=RunMode.FINE))
    with dspy.context(lm=dummy_lm):
        pred = await handle.run(question="color of the sky?")
    assert pred.answer == "blue"


async def _should_not_call(*args, **kwargs):  # pragma: no cover - guard
    raise AssertionError("the wrong execute_* coroutine was dispatched")
