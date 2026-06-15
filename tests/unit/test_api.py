"""Tests for the declare/bind API: program() + the TemporalProgram reference, the
standalone run_program / start_program_nowait client helpers, and the ref's
.start() / .start_nowait() / .result_of().

The client helpers are covered against a fake client (records the call, hands back
a fake handle) so no Temporal server is needed; an end-to-end `ref.start` /
`ref.start_nowait` lives in the integration suite. `.run`'s in-workflow dispatch is
covered by monkeypatching the execute_* coroutines; its outside-a-workflow degrade
runs a real local DSPy call.
"""

import dataclasses
import re

import dspy
import pytest
from pydantic import BaseModel, field_validator

import dspy_temporal as dt
from dspy_temporal.client import prediction_of
from dspy_temporal.coarse import api as api_mod
from dspy_temporal.coarse.api import TemporalProgram
from dspy_temporal.coarse.workflow import DSPyProgramWorkflow
from dspy_temporal.config import CallOptions, RunMode
from dspy_temporal.fine.workflow import DSPyProgramFineWorkflow
from dspy_temporal.models import ProgramCallInput, ProgramCallOutput
from dspy_temporal.registry import default_registry, register_program


class FakeHandle:
    """A stand-in WorkflowHandle whose result() returns a canned output."""

    def __init__(self, output, *, workflow_id="wf-fake"):
        self._output = output
        self.id = workflow_id

    async def result(self):
        return self._output


class FakeClient:
    """Records the start_workflow call and returns a handle over a canned output."""

    def __init__(self):
        self.calls = []

    async def start_workflow(self, run, call, *, id, task_queue, result_type=None):
        self.calls.append(
            {"run": run, "call": call, "id": id, "task_queue": task_queue}
        )
        return FakeHandle(
            ProgramCallOutput(prediction={"answer": "blue"}), workflow_id=id
        )


# --- program(): a pure, immutable reference ----------------------------------


def test_program_is_pure_and_returns_immutable_ref():
    ref = dt.program(
        "pure1",
        mode=RunMode.FINE,
        options=CallOptions(maximum_attempts=7),
        activity_task_queue="gpu",
    )
    assert isinstance(ref, TemporalProgram)
    assert ref.name == "pure1"
    assert ref.mode == RunMode.FINE
    assert ref.options.maximum_attempts == 7
    assert ref.activity_task_queue == "gpu"
    # Declaration mutates nothing: no registry entry until bind().
    assert "pure1" not in default_registry()


def test_program_defaults():
    ref = dt.program("pure2")
    assert ref.mode == RunMode.COARSE
    assert ref.options is None
    assert ref.activity_task_queue is None
    assert ref.result is None


def test_ref_is_frozen():
    ref = dt.program("fz")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.name = "other"


def test_with_options_returns_modified_copy():
    ref = dt.program("wo", mode=RunMode.FINE)
    ref2 = ref.with_options(CallOptions(maximum_attempts=9))

    assert ref2 is not ref
    assert ref2.options.maximum_attempts == 9
    assert ref.options is None  # original untouched
    # All other fields are preserved by the copy.
    assert ref2.name == "wo"
    assert ref2.mode == RunMode.FINE


def test_on_task_queue_returns_modified_copy():
    ref = dt.program("otq")
    ref2 = ref.on_task_queue("gpu-pool")

    assert ref2 is not ref
    assert ref2.activity_task_queue == "gpu-pool"
    assert ref.activity_task_queue is None  # original untouched


# --- bind(): the heavy, side-effecting registration --------------------------


def test_bind_registers_and_returns_self():
    ref = dt.program("b1")
    out = ref.bind(lambda: dspy.Predict("q -> a"))
    assert out is ref  # chainable
    assert "b1" in default_registry()


def test_bind_with_instance_strips_lm_and_records_mode():
    ref = dt.program("b2", mode=RunMode.FINE)
    ref.bind(dspy.ChainOfThought("question -> answer"))

    assert default_registry().mode_for("b2") == RunMode.FINE
    # The registered prototype builds an LM-stripped copy.
    built = default_registry().build("b2")
    assert all(p.lm is None for _n, p in built.named_predictors())


def test_bind_same_object_is_noop_different_raises():
    def builder():
        return dspy.Predict("q -> a")

    dt.program("b3").bind(builder)
    dt.program("b3").bind(builder)  # same object -> no-op
    with pytest.raises(ValueError, match=r"already registered to a different object"):
        dt.program("b3").bind(lambda: dspy.Predict("q -> a"))  # different -> conflict


def test_bind_refused_in_sandbox(monkeypatch):
    """bind() funnels through register_program, so it inherits the sandbox
    guardrail: a top-level bind() in a workflow file (which the sandbox re-execs
    each task) is refused with a RuntimeError pointing at the declare/bind split."""
    from dspy_temporal import registry as registry_mod

    monkeypatch.setattr(
        registry_mod.workflow.unsafe, "in_sandbox", lambda: True, raising=True
    )
    with pytest.raises(RuntimeError, match=r"sandbox"):
        dt.program("sandbox_bind").bind(lambda: dspy.Predict("q -> a"))
    assert "sandbox_bind" not in default_registry()


# --- run_program (low-level by-name path) ------------------------------------


@pytest.mark.asyncio
async def test_run_program_generates_default_workflow_id():
    client = FakeClient()

    pred = await dt.run_program(
        client, "qa", {"question": "sky?"}, task_queue="tq", mode=dt.RunMode.COARSE
    )

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
        mode=dt.RunMode.COARSE,
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


# --- start_program_nowait / prediction_of (by-name non-blocking path) ---------


@pytest.mark.asyncio
async def test_start_program_nowait_returns_handle_and_records_call():
    client = FakeClient()

    handle = await dt.start_program_nowait(
        client, "qa", {"question": "sky?"}, task_queue="tq", mode=RunMode.COARSE
    )

    # Returns the handle itself, not the awaited result.
    assert isinstance(handle, FakeHandle)
    call = client.calls[0]
    assert re.fullmatch(r"dspy-qa-[0-9a-f]{12}", call["id"])
    assert call["task_queue"] == "tq"
    assert isinstance(call["call"], ProgramCallInput)
    assert call["run"] == DSPyProgramWorkflow.run


@pytest.mark.asyncio
async def test_start_program_nowait_selects_fine_workflow_for_fine_mode():
    client = FakeClient()
    await dt.start_program_nowait(
        client, "qa", {"question": "sky?"}, task_queue="tq", mode=RunMode.FINE
    )
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_prediction_of_from_typed_output():
    """A handle whose result() yields a ProgramCallOutput decodes straight back."""
    handle = FakeHandle(ProgramCallOutput(prediction={"answer": "blue"}))
    pred = await prediction_of(handle)
    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_prediction_of_from_raw_dict():
    """A handle re-obtained via get_workflow_handle(id) carries no result type, so
    result() hands back the raw decoded dict -- prediction_of validates it."""
    handle = FakeHandle({"prediction": {"answer": "blue"}, "lm_usage": None})
    pred = await prediction_of(handle)
    assert pred.answer == "blue"


# --- ref.start: standalone start using the ref's mode + an explicit queue -----


@pytest.mark.asyncio
async def test_start_uses_passed_queue_and_default_workflow_id():
    client = FakeClient()
    ref = dt.program("qa")

    pred = await ref.start(client, task_queue="tq-start", question="sky?")

    assert pred.answer == "blue"
    call = client.calls[0]
    assert call["task_queue"] == "tq-start"
    assert re.fullmatch(r"dspy-qa-[0-9a-f]{12}", call["id"])
    assert isinstance(call["call"], ProgramCallInput)
    assert call["call"].program == "qa"
    # Coarse ref -> coarse workflow.
    assert call["run"] != DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_start_requires_task_queue():
    """task_queue is a required keyword on start(): omitting it is a TypeError, not
    a run against a surprise default queue."""
    client = FakeClient()
    ref = dt.program("qa")
    with pytest.raises(TypeError):
        await ref.start(client, question="sky?")


@pytest.mark.asyncio
async def test_start_selects_fine_workflow_from_ref_mode():
    client = FakeClient()
    ref = dt.program("qa", mode=RunMode.FINE)

    await ref.start(client, task_queue="tq", question="sky?")

    # The ref's own mode picks the workflow -- no mode re-pass at the call.
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_start_honors_workflow_id_and_options_overrides():
    client = FakeClient()
    ref = dt.program("qa")
    opts = CallOptions(maximum_attempts=9)

    await ref.start(
        client,
        task_queue="tq",
        question="sky?",
        workflow_id="wf-explicit",
        options=opts,
    )

    call = client.calls[0]
    assert call["id"] == "wf-explicit"
    assert call["call"].options.maximum_attempts == 9


@pytest.mark.asyncio
async def test_start_uses_ref_default_options():
    """A ref-level ``options`` default (``program(..., options=...)``) is honored on
    the start path -- the program's declared timeout/retry, not CallOptions()
    defaults, reaches the standalone workflow when ``start`` omits ``options``."""
    client = FakeClient()
    ref = dt.program("qa", options=CallOptions(maximum_attempts=5))

    await ref.start(client, task_queue="tq", question="sky?")

    assert client.calls[0]["call"].options.maximum_attempts == 5


@pytest.mark.asyncio
async def test_start_options_arg_overrides_ref_default():
    """An explicit ``options`` on ``start`` overrides the ref-level default for
    that one start."""
    client = FakeClient()
    ref = dt.program("qa", options=CallOptions(maximum_attempts=5))

    await ref.start(
        client,
        task_queue="tq",
        question="sky?",
        options=CallOptions(maximum_attempts=9),
    )

    assert client.calls[0]["call"].options.maximum_attempts == 9


@pytest.mark.asyncio
async def test_start_input_named_client_is_not_swallowed():
    """``client`` is positional-only, so a program input field literally named
    ``client`` is forwarded as an input, not bound to start's own parameter."""
    client = FakeClient()
    ref = dt.program("qa")

    await ref.start(client, task_queue="tq", client="acme-corp", question="sky?")

    call = client.calls[0]
    assert call["call"].inputs == {"client": "acme-corp", "question": "sky?"}


# --- ref.start_nowait / ref.result_of: non-blocking start + deferred decode ---


@pytest.mark.asyncio
async def test_start_nowait_returns_handle_without_result_adapter():
    """start_nowait hands back the raw handle -- the result isn't ready, so the
    ref's result adapter is NOT applied here (result_of applies it later)."""
    client = FakeClient()
    ref = dt.program("qa", result=lambda p: _Answer(answer=p.answer))

    handle = await ref.start_nowait(client, task_queue="tq-start", question="sky?")

    assert isinstance(handle, FakeHandle)  # not an _Answer
    call = client.calls[0]
    assert call["task_queue"] == "tq-start"
    assert re.fullmatch(r"dspy-qa-[0-9a-f]{12}", call["id"])
    assert call["call"].program == "qa"
    assert call["run"] != DSPyProgramFineWorkflow.run  # coarse ref -> coarse workflow


@pytest.mark.asyncio
async def test_start_nowait_requires_task_queue():
    """task_queue is a required keyword on start_nowait too: omitting it is a
    TypeError, not a start against a surprise default queue."""
    client = FakeClient()
    ref = dt.program("qa")
    with pytest.raises(TypeError):
        await ref.start_nowait(client, question="sky?")


@pytest.mark.asyncio
async def test_start_nowait_selects_fine_workflow_from_ref_mode():
    client = FakeClient()
    ref = dt.program("qa", mode=RunMode.FINE)

    await ref.start_nowait(client, task_queue="tq", question="sky?")

    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_start_nowait_uses_ref_default_options():
    """A ref-level ``options`` default is honored on the start_nowait path too."""
    client = FakeClient()
    ref = dt.program("qa", options=CallOptions(maximum_attempts=5))

    await ref.start_nowait(client, task_queue="tq", question="sky?")

    assert client.calls[0]["call"].options.maximum_attempts == 5


@pytest.mark.asyncio
async def test_start_nowait_options_arg_overrides_ref_default():
    client = FakeClient()
    ref = dt.program("qa", options=CallOptions(maximum_attempts=5))

    await ref.start_nowait(
        client,
        task_queue="tq",
        question="sky?",
        options=CallOptions(maximum_attempts=9),
    )

    assert client.calls[0]["call"].options.maximum_attempts == 9


@pytest.mark.asyncio
async def test_start_nowait_input_named_client_is_not_swallowed():
    """``client`` is positional-only on start_nowait too, so an input field named
    ``client`` is forwarded as an input."""
    client = FakeClient()
    ref = dt.program("qa")

    await ref.start_nowait(client, task_queue="tq", client="acme-corp", question="sky?")

    assert client.calls[0]["call"].inputs == {"client": "acme-corp", "question": "sky?"}


@pytest.mark.asyncio
async def test_result_of_applies_result_adapter():
    """result_of re-applies the ref's adapter, so a polled handle yields the ref's
    typed value (same contract as start(), deferred to poll time)."""
    client = FakeClient()
    ref = dt.program("qa", result=lambda p: _Answer(answer=p.answer))

    handle = await ref.start_nowait(client, task_queue="tq", question="sky?")
    out = await ref.result_of(handle)

    assert isinstance(out, _Answer)
    assert out.answer == "blue"


@pytest.mark.asyncio
async def test_result_of_without_adapter_returns_prediction():
    client = FakeClient()
    ref = dt.program("qa")

    handle = await ref.start_nowait(client, task_queue="tq", question="sky?")
    out = await ref.result_of(handle)

    assert isinstance(out, dspy.Prediction)
    assert out.answer == "blue"


# --- .run: context-aware dispatch -------------------------------------------


@pytest.mark.asyncio
async def test_run_in_workflow_coarse_dispatches_execute_coarse(monkeypatch):
    recorded = {}

    async def fake_execute_coarse(name, inputs, options, *, task_queue=None):
        recorded["name"] = name
        recorded["inputs"] = inputs
        recorded["options"] = options
        recorded["task_queue"] = task_queue
        return dspy.Prediction(answer="from_coarse")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_coarse", fake_execute_coarse)
    monkeypatch.setattr(api_mod, "execute_fine", _should_not_call)

    ref = dt.program("qa", mode=RunMode.COARSE)
    pred = await ref.run(question="sky?")

    assert pred.answer == "from_coarse"
    assert recorded["name"] == "qa"
    assert recorded["inputs"] == {"question": "sky?"}
    # A bare ref carries no options and co-locates the activity (no task_queue).
    assert recorded["options"] is None
    assert recorded["task_queue"] is None


@pytest.mark.asyncio
async def test_run_in_workflow_coarse_forwards_options_and_task_queue(monkeypatch):
    recorded = {}

    async def fake_execute_coarse(name, inputs, options, *, task_queue=None):
        recorded["options"] = options
        recorded["task_queue"] = task_queue
        return dspy.Prediction(answer="ok")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_coarse", fake_execute_coarse)
    monkeypatch.setattr(api_mod, "execute_fine", _should_not_call)

    opts = CallOptions(maximum_attempts=5)
    ref = dt.program("qa", options=opts).on_task_queue("gpu-pool")
    await ref.run(question="sky?")

    assert recorded["options"] is opts
    assert recorded["task_queue"] == "gpu-pool"


@pytest.mark.asyncio
async def test_run_in_workflow_fine_dispatches_execute_fine(monkeypatch):
    recorded = {}

    async def fake_execute_fine(name, inputs, options, *, task_queue=None):
        recorded["name"] = name
        recorded["options"] = options
        recorded["task_queue"] = task_queue
        return dspy.Prediction(answer="from_fine")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_fine", fake_execute_fine)
    monkeypatch.setattr(api_mod, "execute_coarse", _should_not_call)

    opts = CallOptions(maximum_attempts=4)
    ref = dt.program("qa", mode=RunMode.FINE, options=opts)
    pred = await ref.run(question="sky?")

    assert pred.answer == "from_fine"
    assert recorded["name"] == "qa"
    assert recorded["options"] is opts
    # A bare fine ref co-locates its per-call activities (no task_queue).
    assert recorded["task_queue"] is None


@pytest.mark.asyncio
async def test_run_in_workflow_fine_forwards_task_queue(monkeypatch):
    """activity_task_queue routes fine mode too: execute_fine receives the queue so
    every per-call activity lands on the dedicated pool (not a silent no-op)."""
    recorded = {}

    async def fake_execute_fine(name, inputs, options, *, task_queue=None):
        recorded["task_queue"] = task_queue
        return dspy.Prediction(answer="ok")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_fine", fake_execute_fine)
    monkeypatch.setattr(api_mod, "execute_coarse", _should_not_call)

    ref = dt.program("qa", mode=RunMode.FINE).on_task_queue("gpu-pool")
    await ref.run(question="sky?")

    assert recorded["task_queue"] == "gpu-pool"


@pytest.mark.asyncio
async def test_run_outside_workflow_coarse_runs_in_process(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    ref = dt.program("degrade_coarse", mode=RunMode.COARSE)
    ref.bind(lambda: dspy.ChainOfThought("question -> answer"))

    with dspy.context(lm=dummy_lm):
        pred = await ref.run(question="color of the sky?")
    assert pred.answer == "blue"


@pytest.mark.asyncio
async def test_run_outside_workflow_fine_runs_in_process(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    ref = dt.program("degrade_fine", mode=RunMode.FINE)
    ref.bind(lambda: dspy.ChainOfThought("question -> answer"))

    with dspy.context(lm=dummy_lm):
        pred = await ref.run(question="color of the sky?")
    assert pred.answer == "blue"


# --- typed result adapter ----------------------------------------------------


class _Answer(BaseModel):
    answer: str
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        # Validation lives on the model, so the adapter stays a thin field-lift.
        return max(0.0, min(1.0, v))


@pytest.mark.asyncio
async def test_run_applies_result_adapter_in_workflow(monkeypatch):
    async def fake_execute_coarse(name, inputs, options, *, task_queue=None):
        # Extra fields on the prediction (e.g. reasoning) are simply not lifted.
        return dspy.Prediction(answer="blue", confidence=1.5, reasoning="ignored")

    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(api_mod, "execute_coarse", fake_execute_coarse)
    monkeypatch.setattr(api_mod, "execute_fine", _should_not_call)

    ref = dt.program(
        "qa", result=lambda p: _Answer(answer=p.answer, confidence=p.confidence)
    )
    out = await ref.run(question="?")

    assert isinstance(out, _Answer)
    assert out.answer == "blue"
    assert out.confidence == 1.0  # clamped by the model's field_validator


@pytest.mark.asyncio
async def test_run_applies_result_adapter_on_degrade_path(monkeypatch, dummy_lm):
    monkeypatch.setattr(api_mod.workflow, "in_workflow", lambda: False)
    ref = dt.program("res_degrade", result=lambda p: _Answer(answer=p.answer))
    ref.bind(lambda: dspy.ChainOfThought("question -> answer"))

    with dspy.context(lm=dummy_lm):
        out = await ref.run(question="color of the sky?")

    assert isinstance(out, _Answer)
    assert out.answer == "blue"


@pytest.mark.asyncio
async def test_start_applies_result_adapter():
    """The result adapter holds on the start path too -- a standalone start returns
    the ref's typed value, not a raw dspy.Prediction (same contract as run())."""
    client = FakeClient()
    ref = dt.program("qa", result=lambda p: _Answer(answer=p.answer))

    out = await ref.start(client, task_queue="tq", question="sky?")

    assert isinstance(out, _Answer)
    assert out.answer == "blue"


async def _should_not_call(*args, **kwargs):  # pragma: no cover - guard
    raise AssertionError("the wrong execute_* coroutine was dispatched")


# --- #29: run_program resolves the run mode from the registry ----------------
# Every branch of the resolution: registered-with-mode (omit / equal / conflict),
# registered-without-mode (omit / explicit), and unregistered (omit / explicit).


@pytest.mark.asyncio
async def test_run_program_registered_with_mode_resolves_without_explicit():
    """A name bound FINE runs FINE when called by name with no explicit mode."""
    client = FakeClient()
    dt.program("r29f", mode=RunMode.FINE).bind(lambda: dspy.Predict("q -> a"))
    await dt.run_program(client, "r29f", {"question": "?"}, task_queue="tq")
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_run_program_registered_with_mode_explicit_equal_ok():
    """Passing the matching explicit mode is accepted (no conflict)."""
    client = FakeClient()
    dt.program("r29e", mode=RunMode.COARSE).bind(lambda: dspy.Predict("q -> a"))
    await dt.run_program(
        client, "r29e", {"question": "?"}, task_queue="tq", mode=RunMode.COARSE
    )
    assert client.calls[0]["run"] == DSPyProgramWorkflow.run


@pytest.mark.asyncio
async def test_run_program_registered_with_mode_explicit_conflict_raises():
    """Bound COARSE but called with mode=FINE -> a clear conflict ValueError."""
    client = FakeClient()
    dt.program("r29c", mode=RunMode.COARSE).bind(lambda: dspy.Predict("q -> a"))
    with pytest.raises(ValueError, match=r"registered as mode='coarse'.*'fine'"):
        await dt.run_program(
            client, "r29c", {"question": "?"}, task_queue="tq", mode=RunMode.FINE
        )
    assert client.calls == []  # never dispatched


@pytest.mark.asyncio
async def test_run_program_registered_without_mode_requires_explicit():
    """register_program without a mode + no explicit mode -> ValueError."""
    client = FakeClient()
    register_program("r29nm", lambda: dspy.Predict("q -> a"))  # no mode
    with pytest.raises(ValueError, match=r"registered without a run mode"):
        await dt.run_program(client, "r29nm", {"question": "?"}, task_queue="tq")
    assert client.calls == []


@pytest.mark.asyncio
async def test_run_program_registered_without_mode_explicit_is_trusted():
    """register_program without a mode but an explicit mode given -> trusted."""
    client = FakeClient()
    register_program("r29nm2", lambda: dspy.Predict("q -> a"))  # no mode
    await dt.run_program(
        client, "r29nm2", {"question": "?"}, task_queue="tq", mode=RunMode.FINE
    )
    assert client.calls[0]["run"] == DSPyProgramFineWorkflow.run


@pytest.mark.asyncio
async def test_run_program_unregistered_requires_explicit_mode():
    """An unregistered name with no explicit mode is ambiguous -> ValueError."""
    client = FakeClient()
    with pytest.raises(ValueError, match=r"not registered in this process"):
        await dt.run_program(client, "never_bound", {"q": "?"}, task_queue="tq")
    assert client.calls == []


@pytest.mark.asyncio
async def test_run_program_unregistered_with_explicit_mode_is_trusted():
    """An unregistered name with an explicit mode is the thin-client escape hatch."""
    client = FakeClient()
    await dt.run_program(
        client, "thin_client", {"q": "?"}, task_queue="tq", mode=RunMode.COARSE
    )
    assert client.calls[0]["run"] == DSPyProgramWorkflow.run


def test_bind_records_mode_in_registry():
    """bind stores the ref's mode in the registry (mode_for reads it back)."""
    dt.program("r29dep", mode=RunMode.FINE).bind(lambda: dspy.Predict("q -> a"))
    assert default_registry().mode_for("r29dep") == RunMode.FINE
