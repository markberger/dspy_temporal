"""``DSPyPlugin`` -- a combined client + worker Temporal plugin for DSPy.

Pass it to the **client** -- ``dt.connect(..., plugins=[DSPyPlugin()])`` or
``Client.connect(..., plugins=[DSPyPlugin()])`` -- and it does two things at once:
it installs the pydantic data converter (so our pydantic input/output models
round-trip with type fidelity) and, because it is *also* a
``temporalio.worker.Plugin``, it auto-propagates to any ``Worker`` built from that
client, contributing the four DSPy activities, the two generic workflows, and the
passthrough sandbox runner. It configures a ``Replayer`` the same way::

    client = await dt.connect("localhost:7233", plugins=[dt.DSPyPlugin()])
    worker = Worker(client, task_queue="dspy-temporal")  # DSPy set auto-added

Apply it on the client **or** directly on a ``Worker`` / ``Replayer`` -- not both:
the framework runs a client plugin that is also a worker plugin on the worker too,
so passing it in both places double-applies it (the identity dedup below tolerates
the fixed set, but extra workflows could duplicate). :func:`build_worker`
constructs its ``Worker`` *through* this plugin, so the plugin is the single source
of truth for the worker set; :data:`DSPY_ACTIVITIES` / :data:`DSPY_WORKFLOWS` stay
exported for advanced hand-wiring.

Merge semantics (important): ``Worker.__init__`` folds explicit kwargs into the
config *before* running plugins, so ``configure_worker`` **extends** any
caller-passed ``activities`` / ``workflows`` rather than overwriting them, and
dedups by identity to tolerate accidental double-application. The framework also
pre-populates ``workflow_runner`` with its default ``SandboxedWorkflowRunner`` (so
``setdefault`` would never apply ours); since the DSPy workflows require our
passthrough sandbox, we replace any ``SandboxedWorkflowRunner`` with the DSPy one
(use ``extra_passthrough_modules`` to add your own prefixes), while leaving a
deliberately different runner type (e.g. ``UnsandboxedWorkflowRunner``) untouched.
The data converter is only filled in when the config still carries the framework
default, so a caller's customized converter is respected.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import temporalio.client
import temporalio.service
import temporalio.worker
from temporalio.converter import DataConverter
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from .coarse.activities import run_program_activity
from .coarse.workflow import DSPyProgramWorkflow
from .converter import data_converter
from .fine.activities import describe_lms_activity, lm_call_activity, tool_call_activity
from .fine.workflow import DSPyProgramFineWorkflow
from .sandbox import default_workflow_runner

# Single source of truth for the fixed DSPy worker set, shared with build_worker.
DSPY_ACTIVITIES = (
    run_program_activity,
    describe_lms_activity,
    lm_call_activity,
    tool_call_activity,
)
DSPY_WORKFLOWS = (DSPyProgramWorkflow, DSPyProgramFineWorkflow)


def _dedup_by_identity(items: Iterable) -> list:
    """Order-preserving dedup by object identity (functions/classes)."""
    out: list = []
    seen: set[int] = set()
    for item in items:
        if id(item) not in seen:
            seen.add(id(item))
            out.append(item)
    return out


class DSPyPlugin(temporalio.client.Plugin, temporalio.worker.Plugin):
    """A combined client + worker plugin that contributes the DSPy set.

    Subclasses both ``temporalio.client.Plugin`` and ``temporalio.worker.Plugin``;
    programs register via import side effects into the process-global registry
    (``deploy`` / ``register_program``), so the plugin wires the fixed activity +
    workflow set regardless of which programs are deployed.
    """

    def __init__(
        self,
        *,
        extra_passthrough_modules: tuple[str, ...] = (),
        max_concurrent_activities: int = 100,
        extra_workflows: tuple = (),
    ):
        self._extra_passthrough_modules = tuple(extra_passthrough_modules)
        self._max_concurrent_activities = max_concurrent_activities
        self._extra_workflows = tuple(extra_workflows)

    # --- shared wiring (used by both worker and replayer) --------------------

    def _merged_workflows(self, existing) -> list:
        return _dedup_by_identity(
            [*(existing or []), *DSPY_WORKFLOWS, *self._extra_workflows]
        )

    def _dspy_runner_or_existing(self, runner):
        # The DSPy workflows require our passthrough sandbox. The framework
        # default runner does not pass dspy/litellm/registry through, so replace
        # any SandboxedWorkflowRunner with ours; respect a different runner type.
        if isinstance(runner, SandboxedWorkflowRunner):
            return default_workflow_runner(*self._extra_passthrough_modules)
        return runner

    def _converter_or_existing(self, existing):
        # Fill in the pydantic converter only when the config still carries the
        # framework default (or none); never clobber a caller's custom converter.
        if existing is None or existing is DataConverter.default:
            return data_converter
        return existing

    # --- client side ---------------------------------------------------------

    def configure_client(
        self, config: temporalio.client.ClientConfig
    ) -> temporalio.client.ClientConfig:
        config["data_converter"] = self._converter_or_existing(
            config.get("data_converter")
        )
        return config

    async def connect_service_client(
        self,
        config: temporalio.service.ConnectConfig,
        next,
    ) -> temporalio.service.ServiceClient:
        return await next(config)

    # --- worker side ---------------------------------------------------------

    def configure_worker(
        self, config: temporalio.worker.WorkerConfig
    ) -> temporalio.worker.WorkerConfig:
        config["activities"] = _dedup_by_identity(
            [*(config.get("activities") or []), *DSPY_ACTIVITIES]
        )
        config["workflows"] = self._merged_workflows(config.get("workflows"))
        config["workflow_runner"] = self._dspy_runner_or_existing(
            config.get("workflow_runner")
        )
        # Back the synchronous activities with a thread pool, unless the caller
        # already provided an executor (default is None).
        if config.get("activity_executor") is None:
            config["activity_executor"] = ThreadPoolExecutor(
                max_workers=self._max_concurrent_activities
            )
        return config

    async def run_worker(self, worker, next):
        # The try/finally seam is the documented Temporal plugin pattern: on
        # graceful worker stop we force-flush any registered tracer provider so the
        # last activities' spans aren't lost. The import is core-only (no OTel); the
        # flush is registered by the tracing subpackage's setup_tracing.
        try:
            return await next(worker)
        finally:
            from .config import get_tracing_shutdown

            fn = get_tracing_shutdown()
            if fn is not None:
                fn()

    # --- replayer ------------------------------------------------------------

    def configure_replayer(
        self, config: temporalio.worker.ReplayerConfig
    ) -> temporalio.worker.ReplayerConfig:
        config["workflows"] = self._merged_workflows(config.get("workflows"))
        config["workflow_runner"] = self._dspy_runner_or_existing(
            config.get("workflow_runner")
        )
        config["data_converter"] = self._converter_or_existing(
            config.get("data_converter")
        )
        return config

    def run_replayer(self, replayer, histories, next):
        return next(replayer, histories)
