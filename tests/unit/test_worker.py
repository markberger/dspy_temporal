"""Structural guards for build_worker (no server; Worker constructor is spied).

build_worker now delegates the DSPy worker set to a single ``DSPyPlugin`` (one
assembly path, shared with the plugin entry point). These tests assert it
constructs the ``Worker`` *through* a correctly-configured plugin and adds no
interceptors of its own. The actual wiring the plugin performs is checked by
exercising its ``configure_worker`` hook over a framework-default config.
"""

from concurrent.futures import ThreadPoolExecutor

from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

import dspy_temporal as dt
from dspy_temporal import worker as worker_mod
from dspy_temporal.plugin import DSPY_ACTIVITIES, DSPY_WORKFLOWS, DSPyPlugin


def _spy_worker(monkeypatch):
    captured = {}

    class FakeWorker:
        def __init__(self, client, **kwargs):
            captured["client"] = client
            captured.update(kwargs)

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    return captured


def _only_plugin(captured) -> DSPyPlugin:
    plugins = captured["plugins"]
    assert len(plugins) == 1
    assert isinstance(plugins[0], DSPyPlugin)
    return plugins[0]


def _worker_config_from(plugin: DSPyPlugin) -> dict:
    """Run the plugin's worker hook over a framework-default config (as Worker would)."""
    return plugin.configure_worker(
        {
            "activities": [],
            "workflows": [],
            "workflow_runner": SandboxedWorkflowRunner(),
            "activity_executor": None,
        }
    )


def test_build_worker_delegates_to_plugin_with_no_interceptors(monkeypatch):
    """build_worker wires the DSPy set via a single DSPyPlugin and adds no
    interceptors (tracing's interceptor is client-only; see docs/tracing-design.md)."""
    captured = _spy_worker(monkeypatch)
    dt.build_worker(object(), task_queue="tq")

    assert captured["task_queue"] == "tq"
    # The worker inherits interceptors from the client; build_worker adds none.
    assert "interceptors" not in captured
    # The plugin assembles the sandbox runner + thread-pool executor.
    cfg = _worker_config_from(_only_plugin(captured))
    assert "dspy" in str(cfg["workflow_runner"])
    assert isinstance(cfg["activity_executor"], ThreadPoolExecutor)


def test_build_worker_passes_through_explicit_kwargs(monkeypatch):
    """A caller may still pass interceptors explicitly (advanced/custom setups)."""
    captured = _spy_worker(monkeypatch)
    sentinel = object()
    dt.build_worker(object(), task_queue="tq", interceptors=[sentinel])
    assert captured["interceptors"] == [sentinel]


def test_build_worker_respects_caller_workflow_runner(monkeypatch):
    """A caller-supplied workflow_runner is forwarded to the Worker constructor.
    (That the plugin then leaves a non-default runner untouched is asserted in
    test_plugin.py::test_configure_worker_respects_caller_runner_and_executor.)"""
    captured = _spy_worker(monkeypatch)
    sentinel = object()
    dt.build_worker(object(), task_queue="tq", workflow_runner=sentinel)
    assert captured["workflow_runner"] is sentinel


def test_build_worker_workflows_come_from_shared_constant(monkeypatch):
    """The set the plugin contributes is exactly DSPY_WORKFLOWS + DSPY_ACTIVITIES
    (single source of truth; no inline literals)."""
    captured = _spy_worker(monkeypatch)
    dt.build_worker(object(), task_queue="tq")

    cfg = _worker_config_from(_only_plugin(captured))
    assert cfg["workflows"] == list(DSPY_WORKFLOWS)
    assert cfg["activities"] == list(DSPY_ACTIVITIES)


def test_build_worker_includes_extra_workflows(monkeypatch):
    """A caller's own @workflow.defn classes are appended after the DSPy ones."""
    captured = _spy_worker(monkeypatch)

    class UserWorkflow:
        pass

    dt.build_worker(object(), task_queue="tq", extra_workflows=[UserWorkflow])

    cfg = _worker_config_from(_only_plugin(captured))
    assert cfg["workflows"] == [*DSPY_WORKFLOWS, UserWorkflow]


def test_build_worker_forwards_max_concurrent_and_passthrough(monkeypatch):
    """max_concurrent_activities / extra_passthrough_modules reach the plugin."""
    captured = _spy_worker(monkeypatch)
    dt.build_worker(
        object(),
        task_queue="tq",
        max_concurrent_activities=7,
        extra_passthrough_modules=("my_pkg",),
    )

    cfg = _worker_config_from(_only_plugin(captured))
    assert cfg["activity_executor"]._max_workers == 7
    assert "my_pkg" in str(cfg["workflow_runner"])
