"""Configuration: run config and worker-side LM setup.

The serializable, dspy-free ``CallOptions`` lives in ``options.py`` (re-exported
here) so the workflow can import it without dragging in dspy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import dspy

from .options import DEFAULT_NON_RETRYABLE, CallOptions  # noqa: F401  (re-exported)


@dataclass
class RunConfig:
    """Client/worker-side defaults for deploying and running programs."""

    task_queue: str = "dspy-temporal"
    call_options: CallOptions = field(default_factory=CallOptions)


# --- Worker-side LM configuration -------------------------------------------
# The LM (and its API keys) lives only in the worker process, configured from
# the environment at startup -- never passed through Temporal payloads.

_WORKER_LM: dspy.BaseLM | None = None


def configure_lm_from_env(model: str | None = None, **lm_kwargs) -> dspy.BaseLM:
    """Build a ``dspy.LM`` from the environment and register it as worker LM.

    Reads the model id from the ``DSPY_LM_MODEL`` env var unless ``model`` is
    given. Provider credentials are read by litellm from the usual env vars
    (e.g. ``OPENAI_API_KEY``).
    """
    model = model or os.environ.get("DSPY_LM_MODEL")
    if not model:
        raise ValueError(
            "No LM model configured. Pass model=... or set DSPY_LM_MODEL "
            "(e.g. 'openai/gpt-4o-mini')."
        )
    lm = dspy.LM(model, **lm_kwargs)
    set_worker_lm(lm)
    return lm


def set_worker_lm(lm: dspy.BaseLM) -> None:
    """Set the default LM applied to programs that don't carry their own."""
    global _WORKER_LM
    _WORKER_LM = lm
    # Make it the process-global default too, for threads that don't enter the
    # activity's dspy.context (e.g. background work inside a builder).
    dspy.configure(lm=lm)


def get_worker_lm() -> dspy.BaseLM | None:
    return _WORKER_LM
