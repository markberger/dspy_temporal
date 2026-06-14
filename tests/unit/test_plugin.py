"""Tests for DSPyPlugin: client + worker + replayer configuration.

The key risk on the worker side is the single-source-of-truth / non-clobber
guard: because ``Worker.__init__`` folds explicit kwargs into the config *before*
running plugins, ``configure_worker`` must EXTEND caller-passed
activities/workflows (not overwrite them) and must replace the framework-default
sandbox runner with the DSPy one (``setdefault`` would silently no-op since the
key is always present). The client/replayer hooks must install the pydantic data
converter while respecting a caller-customized one.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio.converter import DataConverter
from temporalio.worker import UnsandboxedWorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from dspy_temporal.converter import data_converter
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
    # Combined client+worker ABC (6 abstract hooks); omitting any makes it
    # uninstantiable. Instantiating proves we implement all of them.
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


def test_configure_client_sets_pydantic_converter_when_default():
    plugin = DSPyPlugin()
    out = plugin.configure_client({"data_converter": DataConverter.default})
    assert out["data_converter"] is data_converter


def test_configure_client_respects_custom_converter():
    plugin = DSPyPlugin()
    custom = object()  # any non-default converter is left untouched
    out = plugin.configure_client({"data_converter": custom})
    assert out["data_converter"] is custom


@pytest.mark.asyncio
async def test_connect_service_client_is_passthrough():
    plugin = DSPyPlugin()
    seen = {}

    async def fake_next(config):
        seen["config"] = config
        return "service-client"

    result = await plugin.connect_service_client("the-config", fake_next)
    assert result == "service-client"
    assert seen["config"] == "the-config"


def test_configure_replayer_adds_workflows_runner_and_converter():
    plugin = DSPyPlugin()
    out = plugin.configure_replayer(
        {
            "workflows": [],
            "workflow_runner": SandboxedWorkflowRunner(),
            "data_converter": DataConverter.default,
        }
    )
    assert out["workflows"] == list(DSPY_WORKFLOWS)
    assert isinstance(out["workflow_runner"], SandboxedWorkflowRunner)
    assert "dspy" in str(out["workflow_runner"])
    assert out["data_converter"] is data_converter


@pytest.mark.asyncio
async def test_plugin_run_worker_is_noop_without_shutdown():
    """With no tracing-shutdown registered, run_worker is a pure passthrough
    (returns next's result, no error)."""
    from dspy_temporal import config as core_config

    core_config.clear_tracing_shutdown()
    plugin = DSPyPlugin()
    seen = {}

    async def fake_next(worker):
        seen["worker"] = worker
        return "ran"

    result = await plugin.run_worker("the-worker", fake_next)
    assert result == "ran"
    assert seen["worker"] == "the-worker"


@pytest.mark.asyncio
async def test_plugin_run_worker_flushes_registered_shutdown_once():
    """A registered tracing-shutdown is invoked exactly once on worker stop, and
    run_worker still returns next's result."""
    from dspy_temporal import config as core_config

    calls = {"n": 0}
    core_config.set_tracing_shutdown(lambda: calls.__setitem__("n", calls["n"] + 1))
    plugin = DSPyPlugin()

    async def fake_next(worker):
        return "ran"

    result = await plugin.run_worker("the-worker", fake_next)
    assert result == "ran"
    assert calls["n"] == 1


def test_plugin_run_replayer_is_passthrough():
    plugin = DSPyPlugin()
    seen = {}

    def fake_next(replayer, histories):
        seen["args"] = (replayer, histories)
        return "replayed"

    result = plugin.run_replayer("replayer", "histories", fake_next)
    assert result == "replayed"
    assert seen["args"] == ("replayer", "histories")
