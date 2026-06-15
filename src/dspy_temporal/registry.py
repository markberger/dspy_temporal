"""Program registry: maps a program *name* to a zero-arg builder or a prototype instance.

Workflows and activity inputs carry only a program name plus call inputs --
never a live ``dspy.Module`` (which would serialize child predictors' LM
objects, including API keys, into durable Temporal history). The worker process
owns the registry and reconstructs a fresh module on demand via its builder.

A caller may register a live ``dspy.Module`` *instance* (e.g. a compiled program
with few-shot demos) instead of a builder; the registry keeps the prototype in
worker memory and mints a fresh, LM-stripped ``deepcopy`` per ``build()`` so the
prototype's demos are preserved while its bound LMs (and their API keys) never
reach a built copy or the Temporal boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import dspy

# temporalio is a hard dependency already, so importing ``workflow`` here is free.
# It's used solely by the module-level register_program sandbox guardrail below
# (registry.py is otherwise deliberately minimal-import); the pure ProgramRegistry
# data structure never touches it.
from temporalio import workflow

# RunMode is the dspy-free run-mode enum from options.py (NOT config.py, which
# imports dspy). Keeping the import here means registry.py never imports dspy
# machinery beyond ``dspy`` itself and never touches the fine/ or coarse/ layers
# -- preserving the one-directional ``fine -> registry`` layering.
from .options import RunMode

ModuleBuilder = Callable[[], dspy.Module]
# Either a zero-arg builder or a live prototype instance the registry clones.
ModuleSource = ModuleBuilder | dspy.Module


def all_named_predictors(module: dspy.Module) -> list[tuple[str, dspy.Predict]]:
    """Like ``module.named_predictors()`` but also reaches predictors inside
    *compiled* sub-modules.

    DSPy's ``named_parameters()`` walk (which ``named_predictors()`` filters) skips
    any sub-module whose ``_compiled`` flag is set -- the flag optimizers stamp on
    a program once they've compiled it. That hides every predictor living inside a
    compiled sub-module, so an LM bound there would survive ``_copy_stripped``
    (leaking its API key into a built copy) and would never receive a per-predictor
    ``WorkflowLM`` in fine mode (its real LM would then run inside the workflow
    sandbox instead of a recorded activity). We temporarily clear ``_compiled`` on
    every sub-module, take the normal walk -- so the returned names keep the bare
    dotted convention the rest of the package keys ``lm_ref`` on (NOT the
    ``self.``-prefixed names ``named_sub_modules`` yields) -- then restore the flags.

    Pure-Python and I/O-free, so it stays safe to call from the fine workflow's
    sandboxed build path. (DSPy 3.2.x: see the ``_compiled`` skip in
    ``primitives/base_module.py:named_parameters`` and the ``skip_compiled=False``
    default of ``named_sub_modules``, which lets us enumerate compiled descendants.)
    """
    compiled = [
        sub
        for _name, sub in module.named_sub_modules(type_=dspy.Module)
        if getattr(sub, "_compiled", False)
    ]
    for sub in compiled:
        sub._compiled = False
    try:
        return module.named_predictors()
    finally:
        for sub in compiled:
            sub._compiled = True


def _copy_stripped(module: dspy.Module) -> dspy.Module:
    """Return a fresh copy of ``module`` with every predictor's ``.lm`` dropped.

    ``deepcopy`` (not ``reset_copy``) so a compiled program's few-shot demos are
    preserved; nulling each predictor's ``.lm`` drops any bound live LM (and its
    API keys) so the prototype's secrets never reach a built copy. Uses
    :func:`all_named_predictors` so predictors inside compiled sub-modules are
    stripped too (else their bound LM, and its key, would survive the clone).
    Pure-Python and I/O-free, so it is safe to call from the fine workflow's
    sandboxed build path.
    """
    clone = module.deepcopy()
    for _name, predictor in all_named_predictors(clone):
        predictor.lm = None
    return clone


@dataclass(frozen=True)
class _Entry:
    """One program's registration record: its builder plus the metadata the
    registry needs to enforce conflict semantics and resolve a run mode.

    - ``builder``: the zero-arg callable that mints a fresh module on ``build()``.
      A prototype instance is normalized into a clone-on-build closure at
      registration time, so an ``_Entry`` only ever holds a builder.
    - ``source``: the *original* object the name was registered with. Conflict
      detection compares by identity: re-registering the **same** object is a
      no-op; a **different** object under a taken name raises.
    - ``mode``: the :class:`RunMode` the name was registered with (via ``ref.bind``
      / ``register_program(..., mode=...)``), or None if registered without one.
    """

    builder: ModuleBuilder
    source: ModuleSource
    mode: RunMode | None


class ProgramRegistry:
    """A name -> builder mapping owned by the worker process.

    A registered prototype instance is normalized at registration time into a
    builder closure that clones it (LM-stripped) on each ``build()``, so the rest
    of the registry only ever deals with builders.

    Each name maps to a single :class:`_Entry` holding ``(builder, source, mode)``:

    - ``source`` is the *original* object the name was registered with, used to
      enforce conflict semantics. Re-registering the **same object** under a name
      is a no-op (a worker that re-imports a module shouldn't error); registering
      a **different object** under an already-taken name raises -- callers must
      :meth:`unregister` first to replace deliberately.
    - ``mode`` is the :class:`RunMode` a name was registered with (via ``ref.bind``
      / ``register_program(..., mode=...)``), or None if registered without one.
      :meth:`mode_for` reads it so the client can resolve a run mode from the
      registry instead of trusting a possibly-mismatched explicit argument.

    Two pieces of process infrastructure sit alongside the entry map:

    - ``_listeners``: invalidation callbacks fired with a program *name* after each
      genuine (re-)registration *or* unregistration. The fine-mode LM-map cache
      subscribes here to evict a stale entry. Kept generic and dspy-free so the
      registry never imports its subscribers.
    - ``_generations``: a per-name counter bumped on every invalidation. A cache
      can stamp the generation it built against and discard a stale build if a
      concurrent re-registration has since bumped it (see the fine-mode cache).
    """

    def __init__(self) -> None:
        # name -> _Entry(builder, source, mode). One record per name (#30, #29).
        self._entries: dict[str, _Entry] = {}
        # Invalidation callbacks, fired with a name on each (re-)registration AND
        # unregistration (#28).
        self._listeners: list[Callable[[str], None]] = []
        # Per-name registration generation, bumped on each invalidation. Lets a
        # cache detect that the entry it built against has since been replaced.
        self._generations: dict[str, int] = {}

    def _invalidate(self, name: str) -> None:
        """Mark ``name`` invalidated: bump its generation and fire the listeners.

        Routed through by both :meth:`register` (on a genuine first/replace
        registration) and :meth:`unregister` (only when the name was present), so
        the generation counter and listener fires stay precise -- never on the
        same-object re-register no-op or an unknown-name unregister.
        """
        self._generations[name] = self._generations.get(name, 0) + 1
        for cb in self._listeners:
            cb(name)

    def generation(self, name: str) -> int:
        """The current registration generation for ``name`` (0 if never seen).

        Bumped on every (re-)registration/unregistration of the name. A cache
        captures this before a slow build and re-checks it under its lock, so a
        build raced by a concurrent re-registration is discarded as stale."""
        return self._generations.get(name, 0)

    def register(
        self, name: str, source: ModuleSource, *, mode: RunMode | None = None
    ) -> None:
        """Register a zero-arg builder *or* a live ``dspy.Module`` prototype.

        A prototype is normalized into a builder that mints a fresh, LM-stripped
        ``deepcopy`` per call (preserving compiled demos, dropping bound LMs).

        Re-registering the **same** ``source`` object under ``name`` is a no-op
        (so a worker re-importing a module that calls ``register_program`` at
        import time doesn't error). Registering a **different** object under an
        already-taken ``name`` raises :class:`ValueError`: call :meth:`unregister`
        first to replace it deliberately. ``mode``, when given, is recorded so the
        client can resolve the run mode from the registry (see :meth:`mode_for`).
        """
        # Conflict check first, before any normalization/mutation, so a same-object
        # re-import returns untouched and a different-object collision can't half-
        # write the entry map.
        existing = self._entries.get(name)
        if existing is not None:
            if existing.source is source:
                return  # same object re-imported (worker reload): no-op
            raise ValueError(
                f"Program {name!r} is already registered to a different object. "
                f"Registered: {sorted(self._entries)}. To replace it deliberately, "
                f"call unregister({name!r}) first, then register the new program."
            )

        if isinstance(source, dspy.Module):
            prototype = source

            def builder(_proto: dspy.Module = prototype) -> dspy.Module:
                return _copy_stripped(_proto)
        elif callable(source):
            builder = source
        else:
            raise TypeError(
                f"Program {name!r} source must be a zero-arg callable returning a "
                f"dspy.Module OR a dspy.Module instance, got "
                f"{type(source).__name__}."
            )

        # Construct + store the _Entry only after source validation succeeds, so a
        # bad-source TypeError above leaves the entry map untouched (a later valid
        # register of the same name then succeeds). The invalidation fires last, so
        # it runs exactly on a genuine first/replace registration -- never on the
        # same-object no-op above (which already returned).
        self._entries[name] = _Entry(builder=builder, source=source, mode=mode)
        self._invalidate(name)

    def unregister(self, name: str) -> None:
        """Remove a program by name. Unknown name is a silent no-op.

        A genuine removal fires the invalidation (bumping the generation and
        listeners) so a cache evicts the stale entry; an unknown name does not, so
        listener-fired counts stay precise."""
        if self._entries.pop(name, None) is not None:
            self._invalidate(name)

    def mode_for(self, name: str) -> RunMode | None:
        """The registered run mode for ``name``, or None if registered without one
        (e.g. via the low-level register_program) or not registered at all."""
        entry = self._entries.get(name)
        return entry.mode if entry is not None else None

    def resolve_mode(self, name: str, explicit: RunMode | None) -> RunMode:
        """Resolve the run mode for a by-name run, validating ``explicit`` against
        the registry. The client's :func:`run_program` delegates here.

        - If ``name`` was deployed in this process with a mode, that mode wins; a
          conflicting ``explicit`` raises (use ``handle.start`` to avoid it).
        - If ``name`` was registered *without* a mode (low-level
          ``register_program``), ``explicit`` is required (none -> raises).
        - If ``name`` is not registered in this process (a thin client that never
          imported the program module), ``explicit`` is required (none -> raises).
        """
        entry = self._entries.get(name)
        if entry is not None:
            registered = entry.mode
            if registered is None:  # registered locally but no mode (register_program)
                if explicit is None:
                    raise ValueError(
                        f"Program {name!r} is registered without a run mode and no "
                        f"mode was given. Pass mode=RunMode.COARSE/FINE, or bind it "
                        f"via a ref declared with a mode "
                        f"(program(name, mode=...).bind(impl))."
                    )
                return explicit
            if explicit is not None and explicit != registered:
                raise ValueError(
                    f"Program {name!r} is registered as mode={registered.value!r} but "
                    f"run_program was called with mode={explicit.value!r}. Use "
                    f"handle.start() (the can't-desync path), or pass the matching "
                    f"mode / omit it."
                )
            return registered
        # Not registered in this process.
        if explicit is None:
            raise ValueError(
                f"Program {name!r} is not registered in this process and no mode was "
                f"given -> ambiguous. Import the program module here, or pass "
                f"mode=RunMode.COARSE/FINE (thin-client escape hatch)."
            )
        return explicit

    def add_invalidation_listener(self, cb: Callable[[str], None]) -> None:
        """Register a callback fired with a program *name* after each successful
        (re-)registration or unregistration. Generic and dspy-free: the registry
        never imports the subscriber (keeps the fine -> registry layering
        one-directional)."""
        self._listeners.append(cb)

    def build(self, name: str) -> dspy.Module:
        try:
            entry = self._entries[name]
        except KeyError:
            raise KeyError(
                f"No program registered under {name!r}. Registered: "
                f"{sorted(self._entries)}. Did the worker bind() the program "
                f"(ref.bind(impl)) before serving?"
            ) from None
        module = entry.builder()
        if not isinstance(module, dspy.Module):
            raise TypeError(
                f"Builder for {name!r} returned {type(module).__name__}, expected a "
                f"dspy.Module."
            )
        return module

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def snapshot(self) -> dict[str, _Entry]:
        """A shallow copy of the entry map, for save/restore around a test.

        Entries are immutable (frozen :class:`_Entry`), so a shallow dict copy is a
        faithful snapshot. The generation map and ``_listeners`` are NOT captured:
        both are process infrastructure that must persist across a restore (the
        cache's eviction hook subscribed once at import; generations only ever
        advance, never roll back)."""
        return dict(self._entries)

    def restore(self, snap: dict[str, _Entry]) -> None:
        """Replace the entry map with a snapshot taken by :meth:`snapshot`."""
        self._entries.clear()
        self._entries.update(snap)


# Process-global default registry. ref.bind()/register_program() populate
# this; the activity reads from it at runtime.
_DEFAULT_REGISTRY = ProgramRegistry()


def default_registry() -> ProgramRegistry:
    return _DEFAULT_REGISTRY


def register_program(
    name: str, source: ModuleSource, *, mode: RunMode | None = None
) -> None:
    """Register a program builder or prototype instance in the global registry.

    Pass ``mode`` to record the program's run mode (``ref.bind`` does this); omit it
    to register without one (the client then requires an explicit mode to run it
    by name). Conflict semantics follow :meth:`ProgramRegistry.register`.

    Refuses to run inside the Temporal workflow sandbox (see the guardrail below).
    """
    # Guardrail against a classic footgun: a top-level ``ref.bind()`` /
    # ``register_program()`` placed in a *workflow file*. Temporal's sandbox
    # re-execs that file on EVERY workflow task for deterministic isolation, so an
    # import-time registration there would rebuild and re-register the program on
    # each task (thrashing the registry + the fine-mode cache). The fix is to
    # declare the program with the side-effect-free ``program(...)`` in a module the
    # workflow file imports, and ``bind()`` the implementation on the worker -- see
    # examples/compose_refs.py (the program() ref) + examples/worker.py (the bind).
    #
    # Use ``workflow.unsafe.in_sandbox()``, NOT ``workflow.in_workflow()``: during
    # the sandbox's module re-exec there is no running workflow, so in_workflow()
    # is False and would never fire. Both are False off the sandbox thread, so a
    # normal host import never trips this.
    if workflow.unsafe.in_sandbox():
        raise RuntimeError(
            f"register_program({name!r}) (or ref.bind()) was called inside the "
            f"Temporal workflow sandbox. The sandbox re-execs your workflow file on "
            f"every task, so a top-level bind()/register_program() in a workflow file "
            f"re-registers the program each task. Declare the program with the "
            f"side-effect-free program(...) in a module the workflow imports, and "
            f"bind() the implementation on the worker -- see examples/compose_refs.py "
            f"(the program() ref) and examples/worker.py (the bind)."
        )
    _DEFAULT_REGISTRY.register(name, source, mode=mode)


def unregister_program(name: str) -> None:
    """Remove a program from the global registry (no-op if absent)."""
    _DEFAULT_REGISTRY.unregister(name)
