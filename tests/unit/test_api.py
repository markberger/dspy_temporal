"""Tests for the auto-wrap API: deploy + the TemporalProgram handle, the
standalone run_program client helper, and the handle's .start().

`run_program` and `handle.start` are covered against a fake client (records the
call) so no Temporal server is needed; an end-to-end `handle.start` lives in the
integration suite. `.run`'s in-workflow dispatch is covered by monkeypatching the
execute_* coroutines; its outside-a-workflow degrade runs a real local DSPy call.
"""

import re

import dspy
import pytest

import dspy_temporal as dt
from dspy_temporal.coarse import api as api_mod
from dspy_temporal.coarse.api import TemporalProgram
from dspy_temporal.config import CallOptions, RunMode
from dspy_temporal.fine.workflow import DSPyProgramFineWorkflow
from dspy_temporal.models import ProgramCallInput, ProgramCallOutput
from dspy_temporal.registry import default_registry


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
    handle = dt.deploy(lambda: dspy.Predict("q -> a"), name="qa2", task_queue="tq2")
    assert isinstance(handle, TemporalProgram)
    assert handle.name == "qa2"
    assert "qa2" in default_registry()
    # mode defaults to coarse; task_queue is carried verbatim on the handle.
    assert handle.mode == RunMode.COARSE
    assert handle.task_queue == "tq2"


def test_deploy_with_instance_registers_and_returns_handle():
    handle = dt.deploy(
        dspy.ChainOfThought("question -> answer"),
        name="inst",
        task_queue="tq-inst",
        mode=RunMode.FINE,
    )
    assert isinstance(handle, TemporalProgram)
    assert "inst" in default_registry()
    # mode + task_queue are carried on the handle (the single source of truth).
    assert handle.mode == RunMode.FINE
    assert handle.task_queue == "tq-inst"
    # The registered prototype builds an LM-stripped copy.
    built = default_registry().build("inst")
    assert all(p.lm is None for _n, p in built.named_predictors())


def test_deploy_requires_task_queue():
    # task_queue is a required keyword: omitting it is a TypeError, never a run
    # against a surprise default queue.
    with pytest.raises(TypeError):
        dt.deploy(lambda: dspy.Predict("q -> a"), name="no_tq")


# --- run_program (low-level by-name path) ------------------------------------


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
    client = FakeClient()
    await dt.run_program(
        client, "qa", {"question": "sky?"}, task_queue="tq", mode=RunMode.FINE
    )
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


# --- handle.start: standalone start using the handle's own mode + queue ------


@pytest.mark.asyncio
async def test_start_uses_handle_queue_and_default_workflow_id():
    client = FakeClient()
    handle = TemporalProgram(name="qa", task_queue="tq-handle")

    pred = await handle.start(client, question="sky?")

    assert pred.answer == "blue"
    call = client.calls[0]
    # The handle is authoritative for the queue -- the caller never re-passes it.
    assert call["task_queue"] == "tq-handle"
    assert re.fullmatch(r"dspy-qa-[0-9a-f]{12}", call["id"])
    assert isinstance(call["call"], ProgramCallInput)
    assert call["call"].program == "qa"
    # Coarse handle -> coarse workflow.
    assert call["run"] != DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_start_selects_fine_workflow_from_handle_mode():
    client = FakeClient()
    handle = TemporalProgram(name="qa", task_queue="tq", mode=RunMode.FINE)

    await handle.start(client, question="sky?")

    # The handle's own mode picks the workflow -- no mode re-pass at the call.
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_start_honors_workflow_id_and_options_overrides():
    client = FakeClient()
    handle = TemporalProgram(name="qa", task_queue="tq")
    opts = CallOptions(maximum_attempts=9)

    await handle.start(client, question="sky?", workflow_id="wf-explicit", options=opts)

    call = client.calls[0]
    assert call["id"] == "wf-explicit"
    assert call["call"].options.maximum_attempts == 9


@pytest.mark.asyncio
async def test_start_input_named_client_is_not_swallowed():
    """``client`` is positional-only, so a program input field literally named
    ``client`` is forwarded as an input, not bound to start's own parameter."""
    client = FakeClient()
    handle = TemporalProgram(name="qa", task_queue="tq")

    await handle.start(client, client="acme-corp", question="sky?")

    call = client.calls[0]
    assert call["call"].inputs == {"client": "acme-corp", "question": "sky?"}


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

    handle = TemporalProgram(name="qa", task_queue="tq", mode=RunMode.COARSE)
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

    handle = TemporalProgram(name="qa", task_queue="tq", mode=RunMode.FINE)
    pred = await handle.run(question="sky?")

    assert pred.answer == "from_fine"
    assert recorded["name"] == "qa"


@pytest.mark.asyncio
async def test_run_outside_workflow_coarse_runs_in_process(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="degrade_coarse",
        task_queue="tq",
    )

    handle = TemporalProgram(
        name="degrade_coarse", task_queue="tq", mode=RunMode.COARSE
    )
    with dspy.context(lm=dummy_lm):
        pred = await handle.run(question="color of the sky?")
    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_run_outside_workflow_fine_runs_in_process(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    dt.deploy(
        lambda: dspy.ChainOfThought("question -> answer"),
        name="degrade_fine",
        task_queue="tq",
    )

    handle = TemporalProgram(name="degrade_fine", task_queue="tq", mode=RunMode.FINE)
    with dspy.context(lm=dummy_lm):
        pred = await handle.run(question="color of the sky?")
    assert pred.answer == "blue"


async def _should_not_call(*args, **kwargs):  # pragma: no cover - guard
    raise AssertionError("the wrong execute_* coroutine was dispatched")
