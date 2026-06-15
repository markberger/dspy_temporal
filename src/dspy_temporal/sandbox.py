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
# The whole `dspy_temporal` package is passed through (by prefix) for two reasons:
#
#  1. The fine workflow reads the host's program registry *in workflow code*
#     (default_registry().build(...)) to reconstruct the module it orchestrates, so
#     `dspy_temporal.registry` must share the host's _DEFAULT_REGISTRY -- a sandbox
#     reload would build an empty one.
#  2. A user's `@workflow.defn` declares its program with a `dt.program(...)`
#     reference imported from a side-effect-free module (e.g. `from .refs import
#     classifier`). That import transitively runs `import dspy_temporal`; pinning
#     the package makes the sandbox reuse the already-imported host package instead
#     of re-running `__init__` (and re-importing dspy/litellm/...) on every task --
#     which is what lets the workflow file import the ref with a *plain* import, no
#     `imports_passed_through()` dance. The user's own workflow module is not under
#     this prefix, so it stays fully sandboxed.
#
# `opentelemetry` is included because, when tracing is enabled, the worker inherits
# the client's TracingInterceptor whose workflow-side component imports OTel inside
# the sandbox; it is otherwise unused in workflow code.
PASSTHROUGH_MODULES = (
    "dspy",
    "litellm",
    "urllib3",
    "requests",
    "httpx",
    "tiktoken",
    "tokenizers",
    "opentelemetry",
    "dspy_temporal",
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
