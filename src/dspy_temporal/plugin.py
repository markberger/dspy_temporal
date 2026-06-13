"""``DSPyPlugin`` -- wire the DSPy activity + workflow set into any Temporal Worker.

An alternative to :func:`build_worker` for callers who already construct their own
``Worker`` and want to add DSPy support declaratively::

    Worker(client, task_queue="dspy-temporal", plugins=[dt.DSPyPlugin()])

The plugin contributes the same four activities, the two generic workflows, and
the DSPy sandbox runner that ``build_worker`` wires by hand -- sharing the
:data:`DSPY_ACTIVITIES` / :data:`DSPY_WORKFLOWS` constants so there is a single
source of truth for the set.

Merge semantics (important): ``Worker.__init__`` folds explicit kwargs into the
config *before* running plugins, so this ``configure_worker`` **extends** any
caller-passed ``activities`` / ``workflows`` rather than overwriting them, and
dedups by identity to tolerate accidental double-application. The framework also
pre-populates ``workflow_runner`` with its default ``SandboxedWorkflowRunner`` (so
``setdefault`` would never apply ours); since the DSPy workflows require our
passthrough sandbox, we replace any ``SandboxedWorkflowRunner`` with the DSPy one
(use ``extra_passthrough_modules`` to add your own prefixes), while leaving a
deliberately different runner type (e.g. ``UnsandboxedWorkflowRunner``) untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from temporalio.worker import Plugin, WorkerConfig
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from .coarse.activities import run_program_activity
from .coarse.workflow import DSPyProgramWorkflow
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


class DSPyPlugin(Plugin):
    """A ``temporalio.worker.Plugin`` that contributes the DSPy worker set.

    ``agent`` is accepted for parity with the competitor's
    ``DSPyPlugin(agent, ...)`` call but is advisory: programs register via import
    side effects into the process-global registry (``deploy`` / ``deploy_module``
    / ``register_program``), so the plugin wires the fixed activity + workflow set
    regardless of whether a handle is passed.
    """

    def __init__(
        self,
        agent=None,
        *,
        extra_passthrough_modules: tuple[str, ...] = (),
        max_concurrent_activities: int = 100,
        extra_workflows: tuple = (),
    ):
        self._agent = agent
        self._extra_passthrough_modules = tuple(extra_passthrough_modules)
        self._max_concurrent_activities = max_concurrent_activities
        self._extra_workflows = tuple(extra_workflows)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config["activities"] = _dedup_by_identity(
            [*(config.get("activities") or []), *DSPY_ACTIVITIES]
        )
        config["workflows"] = _dedup_by_identity(
            [*(config.get("workflows") or []), *DSPY_WORKFLOWS, *self._extra_workflows]
        )
        # The DSPy workflows require our passthrough sandbox. The framework
        # default runner does not pass dspy/litellm/registry through, so replace
        # any SandboxedWorkflowRunner with ours; respect a different runner type.
        if isinstance(config.get("workflow_runner"), SandboxedWorkflowRunner):
            config["workflow_runner"] = default_workflow_runner(
                *self._extra_passthrough_modules
            )
        # Back the synchronous activities with a thread pool, unless the caller
        # already provided an executor (default is None).
        if config.get("activity_executor") is None:
            config["activity_executor"] = ThreadPoolExecutor(
                max_workers=self._max_concurrent_activities
            )
        return config

    async def run_worker(self, worker, next):
        return await next(worker)

    def configure_replayer(self, config):
        return config

    def run_replayer(self, replayer, histories, next):
        return next(replayer, histories)
