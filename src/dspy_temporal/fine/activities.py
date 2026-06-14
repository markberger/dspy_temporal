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

import copy
import threading
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
from ..registry import all_named_predictors, default_registry
from ..serde import _SECRET_KWARGS, _jsonify, decode_lm_kwargs, json_safe

# Sentinel ``lm_ref`` for the worker default LM (predictors with no bound ``.lm``).
DEFAULT_LM_REF = "__default__"

_NO_WORKER_LM = (
    "Fine mode requires a worker LM, but none is configured. Call "
    "dspy_temporal.configure_lm_from_env() (or set_worker_lm(...)) at "
    "worker startup."
)

# ``_SECRET_KWARGS`` (api_key/api_base/base_url) is the single source of truth in
# ``..serde``. ``_spec_for`` keeps its explicit pre-filter below as belt-and-
# suspenders; ``serde.json_safe``/``encode_lm_kwargs`` also drop these keys.

# program name -> {predictor_name: bound .lm or None}. Memoizes the
# predictor->LM mapping so lm_call_activity doesn't rebuild the program every
# call. Only *bound* LMs are cached (stable, builder-defined); predictors with no
# bound LM resolve to the *current* worker default, read fresh each call.
#
# Concurrency: fine-mode activities race in a shared ThreadPoolExecutor, so the
# cache is guarded by ``_LM_MAP_CACHE_LOCK``. Reads take no lock (an atomic
# dict.get on the populated fast path). On a miss the build is serialized per
# program name by a dedicated ``_BUILD_LOCKS[name]`` so concurrent first-callers
# build the map *once* instead of N redundant deepcopies (#8): a thread fetches
# (or creates) that per-name lock under the cache lock, releases the cache lock,
# acquires the build lock, double-checks the cache, and only builds if still
# absent. Before building it stamps the registry's current generation for the
# name; after building it installs the map under the cache lock ONLY IF the
# generation hasn't advanced -- so a build raced by a concurrent re-registration
# (which bumps the generation via the invalidation listener) is discarded as
# stale rather than re-poisoning the cache (#1). The generation stamp is the
# correctness guarantee; the per-name lock is purely the contention fix.
# ``_evict_lm_map_entry`` (subscribed to the registry at import) drops a name's
# entry and bumps its generation on every (re-)registration/unregistration.
_LM_MAP_CACHE: dict[str, dict[str, Any]] = {}
_LM_MAP_CACHE_LOCK = threading.Lock()
# program name -> Lock serializing that name's build (created under the cache lock).
_BUILD_LOCKS: dict[str, threading.Lock] = {}
# program name -> registration generation observed when its cached map was built.
# Used to discard a build that a concurrent re-registration has invalidated.
_LM_MAP_GENERATIONS: dict[str, int] = {}


def clear_lm_map_cache() -> None:
    """Drop the per-program LM map cache, build locks, and generation stamps
    (used by tests for isolation)."""
    with _LM_MAP_CACHE_LOCK:
        _LM_MAP_CACHE.clear()
        _BUILD_LOCKS.clear()
        _LM_MAP_GENERATIONS.clear()


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

    Builds the program on the worker, walks ``all_named_predictors()`` (which also
    reaches predictors inside compiled sub-modules), and for each predictor emits
    an LMSpec for its bound ``.lm`` (else the worker default).
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
        # all_named_predictors so a compiled sub-module's predictor gets its own
        # spec under the same name the workflow binds its WorkflowLM to.
        for name, predictor in all_named_predictors(program):
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

    cache = _LM_MAP_CACHE.get(program_name)  # fast path: atomic read, no lock
    if cache is None:
        cache = _build_lm_map(program_name)
    return cache.get(lm_ref) or worker_lm


def _build_lm_map(program_name: str) -> dict[str, Any]:
    """Build (or fetch a concurrently-built) program's predictor->LM map.

    Serialized per name by ``_BUILD_LOCKS[program_name]`` so concurrent
    first-callers build *once* (#8). Stamps the registry generation before the
    build and installs the map only if that generation still holds, so a build
    raced by a re-registration is discarded as stale rather than re-poisoning the
    cache (#1)."""
    registry = default_registry()
    with _LM_MAP_CACHE_LOCK:
        # Fetch-or-create the per-name build lock under the cache lock (so the lock
        # dict itself stays consistent), then release before acquiring it.
        build_lock = _BUILD_LOCKS.setdefault(program_name, threading.Lock())

    with build_lock:
        # Double-check: another thread may have populated the cache while we waited
        # on the build lock -- if so, reuse its map and skip the redundant build.
        cached = _LM_MAP_CACHE.get(program_name)
        if cached is not None:
            return cached

        # Capture the generation BEFORE building: a re-registration during the
        # build bumps it (via _evict_lm_map_entry), marking this build stale.
        generation = registry.generation(program_name)
        program = registry.build(program_name)
        # all_named_predictors so a compiled sub-module's lm_ref resolves to its
        # bound .lm (keys must match describe_lms_activity / the workflow binding).
        built = {
            name: getattr(p, "lm", None) for name, p in all_named_predictors(program)
        }

        with _LM_MAP_CACHE_LOCK:
            # Install only if no concurrent re-registration advanced the generation;
            # otherwise this build is stale -- return it for THIS call but don't
            # poison the cache (the next call rebuilds against the new generation).
            if registry.generation(program_name) == generation:
                _LM_MAP_CACHE[program_name] = built
                _LM_MAP_GENERATIONS[program_name] = generation
            return built


@activity.defn(name="dspy_lm_call")
def lm_call_activity(call: LMCallInput) -> LMCallOutput:
    base = _resolve_lm(call.program, call.lm_ref)

    # Shallow-clone the resolved LM but give it a FRESH, private history, so
    # ``history[-1]`` is unambiguously this call's even when many activities race
    # in the shared ThreadPoolExecutor. We avoid ``base.copy()`` (a full deepcopy)
    # because it's needless per call: the forward path reads ``self.kwargs`` /
    # ``self.callbacks`` only via ``{**self.kwargs, **kwargs}`` and never mutates
    # them in place, so sharing those by reference is safe. The per-call OTel span
    # attribution in ``tracing/callback.py`` DEPENDS on this one-entry history (it
    # drops attribution when ``len(new_entries) != 1``) -- do NOT "optimize" the
    # ``history = []`` reset away.
    lm = copy.copy(base)
    lm.history = []

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


def _evict_lm_map_entry(name: str) -> None:
    """Drop one program's cached LM map + its build-time generation stamp
    (registry invalidation listener).

    Fired by the registry's ``_invalidate`` *after* it has already bumped the
    name's registration generation, so a build in flight for this name (which
    captured the pre-bump generation) will see the advance under the cache lock
    and discard itself as stale instead of re-poisoning the cache (#1). We also
    drop the local generation stamp here since the map it described is gone."""
    with _LM_MAP_CACHE_LOCK:
        _LM_MAP_CACHE.pop(name, None)
        _LM_MAP_GENERATIONS.pop(name, None)


# Subscribe at import so any (re-)registration of a program name evicts its stale
# LM map. The registry stays dspy-free (it never imports this module); the
# dependency points one way, fine -> registry.
default_registry().add_invalidation_listener(_evict_lm_map_entry)
