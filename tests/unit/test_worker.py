"""Structural guards for build_worker (no server; Worker constructor is spied)."""

from concurrent.futures import ThreadPoolExecutor

import dspy_temporal as dt
from dspy_temporal import worker as worker_mod
from dspy_temporal.config import RunConfig
from dspy_temporal.plugin import DSPY_ACTIVITIES, DSPY_WORKFLOWS


def test_build_worker_does_not_register_interceptors(monkeypatch):
    """Enforce the client-only registration rule by construction.

    Tracing's TracingInterceptor must be registered on the CLIENT only; if
    build_worker also added one, spans would double-emit (see
    docs/tracing-design.md). This guards against a future regression that wires
    an interceptor into the worker.
    """
    captured = {}

    class FakeWorker:
        def __init__(self, client, **kwargs):
            captured["client"] = client
            captured.update(kwargs)

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    dt.build_worker(object(), config=RunConfig(task_queue="tq"))

    # The worker inherits interceptors from the client; build_worker adds none.
    assert "interceptors" not in captured
    assert captured["task_queue"] == "tq"
    assert "workflow_runner" in captured  # sandbox runner is wired in
    assert isinstance(captured["activity_executor"], ThreadPoolExecutor)


def test_build_worker_passes_through_explicit_kwargs(monkeypatch):
    """A caller may still pass interceptors explicitly (advanced/custom setups)."""
    captured = {}

    class FakeWorker:
        def __init__(self, client, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    sentinel = object()
    dt.build_worker(
        object(), config=RunConfig(task_queue="tq"), interceptors=[sentinel]
    )
    assert captured["interceptors"] == [sentinel]


def test_build_worker_respects_caller_workflow_runner(monkeypatch):
    """A caller-supplied workflow_runner is not overridden by the sandbox default."""
    captured = {}

    class FakeWorker:
        def __init__(self, client, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    sentinel = object()
    dt.build_worker(
        object(), config=RunConfig(task_queue="tq"), workflow_runner=sentinel
    )
    assert captured["workflow_runner"] is sentinel


def test_build_worker_workflows_come_from_shared_constant(monkeypatch):
    """build_worker registers exactly the shared DSPY_WORKFLOWS + DSPY_ACTIVITIES
    constants (single source of truth with the plugin), no inline literals."""
    captured = {}

    class FakeWorker:
        def __init__(self, client, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    dt.build_worker(object(), config=RunConfig(task_queue="tq"))

    assert captured["workflows"] == list(DSPY_WORKFLOWS)
    assert captured["activities"] == list(DSPY_ACTIVITIES)


def test_build_worker_includes_extra_workflows(monkeypatch):
    """A caller's own @workflow.defn classes are appended after the DSPy ones."""
    captured = {}

    class FakeWorker:
        def __init__(self, client, **kwargs):
            captured.update(kwargs)

    class UserWorkflow:
        pass

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    dt.build_worker(
        object(), config=RunConfig(task_queue="tq"), extra_workflows=[UserWorkflow]
    )

    assert captured["workflows"] == [*DSPY_WORKFLOWS, UserWorkflow]
