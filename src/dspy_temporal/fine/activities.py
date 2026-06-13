"""Fine-mode activities: the actual I/O, one call per activity.

These run *outside* the workflow sandbox (real DSPy, real HTTP), mirroring the
coarse activity's patterns: apply the worker LM / tracing callback via
``dspy.context``, never ``dspy.configure``. Each call is its own activity, so a
crash + replay resumes from the last completed one instead of re-running the
whole program.

The key fix over coarse mode: ``lm_call_activity`` runs on an **isolated copy**
of the worker LM, so ``history[-1]`` reliably belongs to *this* call even when
many activities race in the shared ``ThreadPoolExecutor`` -- eliminating the
coarse token-attribution-under-concurrency caveat.
"""

from __future__ import annotations

import dspy
from temporalio import activity

from ..config import get_tracing_callback, get_worker_lm
from ..models import LMCallInput, LMCallOutput, ToolCallInput, ToolCallOutput
from ..registry import default_registry
from ..serde import _jsonify


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


@activity.defn(name="dspy_lm_call")
def lm_call_activity(call: LMCallInput) -> LMCallOutput:
    worker_lm = get_worker_lm()
    if worker_lm is None:
        raise RuntimeError(
            "Fine mode requires a worker LM, but none is configured. Call "
            "dspy_temporal.configure_lm_from_env() (or set_worker_lm(...)) at "
            "worker startup."
        )

    # copy() => an isolated dspy.LM with its own (empty) history, so history[-1]
    # is unambiguously this call's even under concurrent activities.
    lm = worker_lm.copy()

    with dspy.context(**_with_tracing_callback({})):
        outputs = lm(prompt=call.prompt, messages=call.messages, **call.lm_kwargs)

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
    with dspy.context(**ctx):
        result = tool(**call.args)

    return ToolCallOutput(observation=_jsonify(result))
