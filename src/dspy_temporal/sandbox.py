"""Workflow sandbox configuration.

The coarse workflow imports our package, which (via ``__init__``) pulls in dspy
and its transitive deps (litellm, urllib3, ...). Those are non-deterministic
I/O libraries that must never run *in* a workflow -- but they are only ever used
from inside activities. We mark them as passthrough so the sandbox reuses the
already-imported host modules instead of reloading them and tripping its
restriction checks. Workflow code itself stays sandboxed.
"""

from __future__ import annotations

from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

# Heavy, I/O-bound modules that are activity-only. Passthrough applies by prefix.
# `opentelemetry` is included because, when tracing is enabled, the worker
# inherits the client's TracingInterceptor whose workflow-side component imports
# OTel inside the sandbox; it is otherwise unused in workflow code.
#
# `dspy_temporal.registry` is passed through because the fine workflow reads the
# host's program registry *in workflow code* (default_registry().build(...)) to
# reconstruct the module it orchestrates. It must share the host's
# _DEFAULT_REGISTRY -- a sandbox reload would build an empty one, and the
# workflow's own imports_passed_through() block is too late: importing the
# dspy_temporal package re-runs __init__, which imports .registry fresh before
# that block runs. Pinning it here makes the sandbox importer reuse the host
# module on *every* import, regardless of ordering.
PASSTHROUGH_MODULES = (
    "dspy",
    "litellm",
    "urllib3",
    "requests",
    "httpx",
    "tiktoken",
    "tokenizers",
    "opentelemetry",
    "dspy_temporal.registry",
)


def default_restrictions(*extra_passthrough_modules: str) -> SandboxRestrictions:
    """Sandbox restrictions with the activity-only modules passed through.

    ``extra_passthrough_modules`` adds caller-supplied prefixes -- the fine-mode
    escape hatch for a builder that must reference a module whose import-time
    side effects would otherwise trip the sandbox (see ``build_worker``).
    """
    return SandboxRestrictions.default.with_passthrough_modules(
        *PASSTHROUGH_MODULES, *extra_passthrough_modules
    )


def default_workflow_runner(*extra_passthrough_modules: str) -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=default_restrictions(*extra_passthrough_modules)
    )
