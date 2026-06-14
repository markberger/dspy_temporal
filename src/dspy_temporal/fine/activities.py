"""Fine-mode activities: the actual I/O, one call per activity.

These run *outside* the workflow sandbox (real DSPy, real HTTP), mirroring the
coarse activity's patterns: apply the worker LM / tracing callback via
``dspy.context``, never ``dspy.configure``. Each call is its own activity, so a
crash + replay resumes from the last completed one instead of re-running the
whole program.

The key fix over coarse mode: ``lm_call_activity`` runs on an **isolated copy**
of the resolved LM, so ``history[-1]`` reliably belongs to *this* call even when
many activities race in the shared ``ThreadPoolExecutor`` -- eliminating the
coarse token-attribution-under-concurrency caveat.

``describe_lms_activity`` runs once at workflow start: it introspects each
predictor's *effective* LM (its bound ``.lm`` else the worker default) and
returns a JSON-native :class:`LMSpec` per predictor, so the workflow can stand in
a faithful ``WorkflowLM`` for each. ``lm_call_activity`` then resolves the real LM
by ``lm_ref`` (the predictor name) -- honoring per-predictor multi-LM programs
without ever putting an LM or its credentials on the wire.
"""

from __future__ import annotations

from typing import Any

import dspy
from temporalio import activity

from ..config import get_tracing_callback, get_worker_lm
from ..heartbeat import heartbeating
from ..models import (
    LMCallInput,
    LMCallOutput,
    LMDescribeInput,
    LMSpec,
    LMSpecsOutput,
    ToolCallInput,
    ToolCallOutput,
)
from ..registry import default_registry
from ..serde import _jsonify, decode_lm_kwargs, json_safe

# Sentinel ``lm_ref`` for the worker default LM (predictors with no bound ``.lm``).
DEFAULT_LM_REF = "__default__"

_NO_WORKER_LM = (
    "Fine mode requires a worker LM, but none is configured. Call "
    "dspy_temporal.configure_lm_from_env() (or set_worker_lm(...)) at "
    "worker startup."
)

# Credentials never belong in an LMSpec (it crosses to the workflow). litellm
# reads keys from env at call time, so dropping these is safe.
_SECRET_KWARGS = ("api_key", "api_base", "base_url")

# program name -> {predictor_name: bound .lm or None}. Memoizes the
# predictor->LM mapping so lm_call_activity doesn't rebuild the program every
# call. Only *bound* LMs are cached (stable, builder-defined); predictors with no
# bound LM resolve to the *current* worker default, read fresh each call.
_LM_MAP_CACHE: dict[str, dict[str, Any]] = {}


def clear_lm_map_cache() -> None:
    """Drop the per-program LM map cache (used by tests for isolation)."""
    _LM_MAP_CACHE.clear()


def _with_tracing_callback(ctx_kwargs: dict) -> dict:
    """Attach the tracing callback (if any), deduped against dspy.settings.

    Mirrors the coarse activity's guard: a user may also have registered this
    same callback globally via ``dspy.settings.callbacks``, and a double-add
    would double-emit every span.
    """
    callback = get_tracing_callback()
    if callback is not None:
        callbacks = list(dspy.settings.callbacks or [])
        if callback not in callbacks:
            callbacks.append(callback)
        ctx_kwargs["callbacks"] = callbacks
    return ctx_kwargs


def _spec_for(lm: dspy.BaseLM) -> LMSpec:
    """Describe an LM as a JSON-native LMSpec (no credentials)."""
    kwargs = {
        k: v
        for k, v in dict(getattr(lm, "kwargs", {})).items()
        if k not in _SECRET_KWARGS
    }
    return LMSpec(
        model=lm.model,
        model_type=getattr(lm, "model_type", "chat"),
        supported_params=sorted(lm.supported_params),
        supports_response_schema=lm.supports_response_schema,
        supports_function_calling=lm.supports_function_calling,
        kwargs=json_safe(kwargs),
    )


@activity.defn(name="dspy_describe_lms")
def describe_lms_activity(call: LMDescribeInput) -> LMSpecsOutput:
    """Describe each predictor's effective LM for the workflow.

    Builds the program on the worker, walks ``named_predictors()``, and for each
    predictor emits an LMSpec for its bound ``.lm`` (else the worker default).
    The ``"__default__"`` entry describes the worker LM, used for the workflow's
    context fallback and any predictor created dynamically at call time.
    """
    # Heartbeat while we build the program + introspect predictors, so the
    # option is honored identically across all activities (no-op by default).
    with heartbeating():
        worker_lm = get_worker_lm()
        if worker_lm is None:
            raise RuntimeError(_NO_WORKER_LM)

        program = default_registry().build(call.program)
        specs = {DEFAULT_LM_REF: _spec_for(worker_lm)}
        for name, predictor in program.named_predictors():
            lm = getattr(predictor, "lm", None) or worker_lm
            specs[name] = _spec_for(lm)
        return LMSpecsOutput(specs=specs)


def _resolve_lm(program_name: str | None, lm_ref: str | None) -> dspy.BaseLM:
    """Resolve the LM to run for a call: a predictor's bound .lm, else worker LM."""
    worker_lm = get_worker_lm()
    if worker_lm is None:
        raise RuntimeError(_NO_WORKER_LM)
    if not program_name or not lm_ref or lm_ref == DEFAULT_LM_REF:
        return worker_lm

    cache = _LM_MAP_CACHE.get(program_name)
    if cache is None:
        program = default_registry().build(program_name)
        cache = {name: getattr(p, "lm", None) for name, p in program.named_predictors()}
        _LM_MAP_CACHE[program_name] = cache
    return cache.get(lm_ref) or worker_lm


@activity.defn(name="dspy_lm_call")
def lm_call_activity(call: LMCallInput) -> LMCallOutput:
    base = _resolve_lm(call.program, call.lm_ref)

    # copy() => an isolated dspy.LM with its own (empty) history, so history[-1]
    # is unambiguously this call's even under concurrent activities.
    lm = base.copy()

    with heartbeating(), dspy.context(**_with_tracing_callback({})):
        outputs = lm(
            prompt=call.prompt,
            messages=call.messages,
            **decode_lm_kwargs(call.lm_kwargs),
        )

    entry = lm.history[-1] if lm.history else {}
    return LMCallOutput(
        outputs=outputs,
        usage=entry.get("usage") or {},
        cost=entry.get("cost"),
        model=lm.model,
        response_model=entry.get("response_model"),
    )


@activity.defn(name="dspy_tool_call")
def tool_call_activity(call: ToolCallInput) -> ToolCallOutput:
    program = default_registry().build(call.program)
    tools = getattr(program, "tools", None)
    if not isinstance(tools, dict) or call.tool_name not in tools:
        raise KeyError(
            f"Program {call.program!r} has no tool named {call.tool_name!r}. "
            f"Fine mode resolves tools via program.tools (ReAct and any module "
            f"exposing a .tools dict)."
        )
    tool = tools[call.tool_name]

    # allow_tool_async_sync_conversion lets dspy.Tool run async tool functions
    # from this synchronous activity (it drives them to completion internally).
    ctx = _with_tracing_callback({"allow_tool_async_sync_conversion": True})
    with heartbeating(), dspy.context(**ctx):
        result = tool(**call.args)

    return ToolCallOutput(observation=_jsonify(result))
