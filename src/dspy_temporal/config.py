"""Configuration: run config and worker-side LM setup.

The serializable, dspy-free ``CallOptions`` lives in ``options.py`` (re-exported
here) so the workflow can import it without dragging in dspy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import dspy

from .options import (  # noqa: F401  (re-exported)
    DEFAULT_NON_RETRYABLE,
    CallOptions,
    RunMode,
)


@dataclass
class RunConfig:
    """Client/worker-side defaults for deploying and running programs.

    ``mode`` selects how a deployed program runs:

    - ``RunMode.COARSE`` (default): the whole ``dspy.Module`` runs in one
      activity; durability is job-level (a crash re-runs the whole program).
    - ``RunMode.FINE``: each LM call and each tool call is its own activity,
      orchestrated by the workflow, so completed steps survive a crash and are
      not re-run. See :mod:`dspy_temporal.fine`.

    The mode is consumed when starting a run (it picks the workflow); the worker
    serves both modes, so a worker's ``RunConfig.mode`` does not constrain it.
    """

    task_queue: str = "dspy-temporal"
    mode: RunMode = RunMode.COARSE


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
    # Also set the process-global default. This is the once-at-startup entry
    # point (main thread / the worker's startup task), so dspy.configure is
    # allowed here; set_worker_lm itself stays configure-free so it can be
    # called from any thread or async task.
    dspy.configure(lm=lm)
    return lm


def set_worker_lm(lm: dspy.BaseLM) -> None:
    """Set the default LM applied to programs that don't carry their own.

    The coarse activity applies this LM via ``dspy.context`` per call, so this
    setter deliberately does not touch ``dspy.settings`` -- keeping it safe to
    call from any thread or async task.
    """
    global _WORKER_LM
    _WORKER_LM = lm


def get_worker_lm() -> dspy.BaseLM | None:
    return _WORKER_LM


def clear_worker_lm() -> None:
    """Clear the worker LM so programs fall back to their own / global LM.

    Does not touch ``dspy.settings`` (clearing the global config is not
    supported mid-process); it only drops this module's default override.
    """
    global _WORKER_LM
    _WORKER_LM = None


# --- Program execution: prefer DSPy's async path ----------------------------
# Running a program via ``acall`` (not the sync ``__call__``) is what makes
# in-program concurrency traceable: ``asyncio.gather`` over ``.acall`` copies the
# contextvar context into each Task (PEP 567), so DSPy's ``ACTIVE_CALL_ID``
# propagates and the tracing callback nests spans correctly. ``dspy.Parallel``'s
# ``ThreadPoolExecutor`` does NOT copy contextvars, so it orphans spans -- prefer
# async concurrency when you want a correct trace tree (see docs/tracing-design.md).


def supports_async(program) -> bool:
    """True if ``program`` implements DSPy's async path (``aforward``).

    Base ``dspy.Module`` defines no ``aforward``; built-ins (``Predict`` /
    ``ChainOfThought`` / ``ReAct``) and async-aware custom modules do. The check is
    side-effect-free, so we can pick the path without partially running the program.
    """
    return hasattr(type(program), "aforward")


async def run_program_async_or_sync(program, inputs: dict):
    """Run ``program`` on its async path when available, else synchronously.

    Prefer ``acall`` so concurrent sub-calls (``asyncio.gather`` over ``.acall``)
    nest correctly in traces; modules that implement only ``forward`` fall back to
    the synchronous call. Both paths wrap the run in ``track_usage`` (DSPy does this
    inside ``__call__``/``acall``), so ``prediction.get_lm_usage()`` works either way.
    """
    if supports_async(program):
        return await program.acall(**inputs)
    return program(**inputs)


# --- Worker-side tracing callback -------------------------------------------
# Mirrors the worker-LM accessor pattern above (module-global + set/get/clear).
# Holds the optional DSPy tracing callback as a plain object reference. Kept here
# (not in the tracing subpackage) so the activity can read it WITHOUT importing
# OpenTelemetry into the core import path. The object is a dspy.BaseCallback;
# typed loosely to avoid any OTel import here.

_TRACING_CALLBACK = None


def set_tracing_callback(callback) -> None:
    """Register the DSPy tracing callback applied by the program activity."""
    global _TRACING_CALLBACK
    _TRACING_CALLBACK = callback


def get_tracing_callback():
    return _TRACING_CALLBACK


def clear_tracing_callback() -> None:
    global _TRACING_CALLBACK
    _TRACING_CALLBACK = None
