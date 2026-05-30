"""Workflow sandbox configuration.

The coarse workflow imports our package, which (via ``__init__``) pulls in dspy
and its transitive deps (litellm, urllib3, ...). Those are non-deterministic
I/O libraries that must never run *in* a workflow -- but they are only ever used
from inside activities. We mark them as passthrough so the sandbox reuses the
already-imported host modules instead of reloading them and tripping its
restriction checks. Workflow code itself stays sandboxed.
"""

from __future__ import annotations

from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

# Heavy, I/O-bound modules that are activity-only. Passthrough applies by prefix.
PASSTHROUGH_MODULES = (
    "dspy",
    "litellm",
    "urllib3",
    "requests",
    "httpx",
    "tiktoken",
    "tokenizers",
)


def default_restrictions() -> SandboxRestrictions:
    return SandboxRestrictions.default.with_passthrough_modules(*PASSTHROUGH_MODULES)


def default_workflow_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(restrictions=default_restrictions())
