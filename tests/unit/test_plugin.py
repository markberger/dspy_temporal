"""Tests for DSPyPlugin.configure_worker merge semantics + the abstract hooks.

The key risk is the single-source-of-truth / non-clobber guard: because
``Worker.__init__`` folds explicit kwargs into the config *before* running
plugins, ``configure_worker`` must EXTEND caller-passed activities/workflows (not
overwrite them) and must replace the framework-default sandbox runner with the
DSPy one (``setdefault`` would silently no-op since the key is always present).
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio.worker import UnsandboxedWorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from dspy_temporal.plugin import DSPY_ACTIVITIES, DSPY_WORKFLOWS, DSPyPlugin


def _default_config():
    """Mimic the config Worker.__init__ hands a plugin: defaults already folded in
    (workflow_runner is the framework default SandboxedWorkflowRunner; executor None)."""
    return {
        "activities": [],
        "workflows": [],
        "workflow_runner": SandboxedWorkflowRunner(),
        "activity_executor": None,
    }


def test_plugin_instantiable():
    # Plugin is a 4-method ABC; omitting any method makes it uninstantiable.
    assert isinstance(DSPyPlugin(), DSPyPlugin)


def test_configure_worker_extends_activities_and_workflows():
    plugin = DSPyPlugin()
    config = _default_config()
    config["activities"] = ["caller_activity"]
    config["workflows"] = ["CallerWorkflow"]

    out = plugin.configure_worker(config)

    # Caller's entries are preserved and ours appended (never clobbered).
    assert out["activities"] == ["caller_activity", *DSPY_ACTIVITIES]
    assert out["workflows"] == ["CallerWorkflow", *DSPY_WORKFLOWS]


def test_configure_worker_dedups_on_double_application():
    """Applying the plugin twice does not duplicate the fixed set (identity dedup)."""
    plugin = DSPyPlugin()
    config = plugin.configure_worker(_default_config())
    out = plugin.configure_worker(config)

    assert out["activities"] == list(DSPY_ACTIVITIES)
    assert out["workflows"] == list(DSPY_WORKFLOWS)


def test_configure_worker_appends_extra_workflows():
    class UserWorkflow:
        pass

    plugin = DSPyPlugin(extra_workflows=[UserWorkflow])
    out = plugin.configure_worker(_default_config())

    assert out["workflows"] == [*DSPY_WORKFLOWS, UserWorkflow]


def test_configure_worker_sets_runner_and_executor_when_absent():
    plugin = DSPyPlugin(max_concurrent_activities=7)
    out = plugin.configure_worker(_default_config())

    # The framework-default sandbox runner is replaced with the DSPy one
    # (carrying our passthrough modules), and the activity executor is created.
    assert isinstance(out["workflow_runner"], SandboxedWorkflowRunner)
    assert "dspy" in str(out["workflow_runner"])
    assert isinstance(out["activity_executor"], ThreadPoolExecutor)
    assert out["activity_executor"]._max_workers == 7


def test_configure_worker_respects_caller_runner_and_executor():
    plugin = DSPyPlugin()
    caller_runner = UnsandboxedWorkflowRunner()  # a deliberately different type
    caller_executor = ThreadPoolExecutor(max_workers=1)
    config = _default_config()
    config["workflow_runner"] = caller_runner
    config["activity_executor"] = caller_executor

    out = plugin.configure_worker(config)

    assert out["workflow_runner"] is caller_runner
    assert out["activity_executor"] is caller_executor


def test_configure_worker_passes_extra_passthrough_modules():
    plugin = DSPyPlugin(extra_passthrough_modules=("my_pkg",))
    out = plugin.configure_worker(_default_config())
    assert "my_pkg" in str(out["workflow_runner"])


@pytest.mark.asyncio
async def test_plugin_run_worker_is_passthrough():
    plugin = DSPyPlugin()
    seen = {}

    async def fake_next(worker):
        seen["worker"] = worker
        return "ran"

    result = await plugin.run_worker("the-worker", fake_next)
    assert result == "ran"
    assert seen["worker"] == "the-worker"


def test_plugin_replayer_hooks_are_passthrough():
    plugin = DSPyPlugin()
    cfg = {"some": "config"}
    # configure_replayer returns the config unchanged.
    assert plugin.configure_replayer(cfg) is cfg

    # run_replayer forwards to next(replayer, histories) unchanged.
    seen = {}

    def fake_next(replayer, histories):
        seen["args"] = (replayer, histories)
        return "replayed"

    result = plugin.run_replayer("replayer", "histories", fake_next)
    assert result == "replayed"
    assert seen["args"] == ("replayer", "histories")


def test_plugin_agent_arg_is_advisory():
    # An agent handle may be passed for parity; it does not change wiring.
    plugin = DSPyPlugin(agent=object())
    out = plugin.configure_worker(_default_config())
    assert out["workflows"] == list(DSPY_WORKFLOWS)
