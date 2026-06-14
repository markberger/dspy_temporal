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

import dspy

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


class ProgramRegistry:
    """A name -> builder mapping owned by the worker process.

    A registered prototype instance is normalized at registration time into a
    builder closure that clones it (LM-stripped) on each ``build()``, so the rest
    of the registry only ever deals with builders.
    """

    def __init__(self) -> None:
        self._builders: dict[str, ModuleBuilder] = {}

    def register(self, name: str, source: ModuleSource) -> None:
        """Register a zero-arg builder *or* a live ``dspy.Module`` prototype.

        A prototype is normalized into a builder that mints a fresh, LM-stripped
        ``deepcopy`` per call (preserving compiled demos, dropping bound LMs).
        """
        if isinstance(source, dspy.Module):
            prototype = source

            def builder(_proto: dspy.Module = prototype) -> dspy.Module:
                return _copy_stripped(_proto)

            self._builders[name] = builder
        elif callable(source):
            self._builders[name] = source
        else:
            raise TypeError(
                f"Program {name!r} source must be a zero-arg callable returning a "
                f"dspy.Module OR a dspy.Module instance, got "
                f"{type(source).__name__}."
            )

    def build(self, name: str) -> dspy.Module:
        try:
            builder = self._builders[name]
        except KeyError:
            raise KeyError(
                f"No program registered under {name!r}. Registered: "
                f"{sorted(self._builders)}. Did the worker import the module that "
                f"calls deploy()/register_program()?"
            ) from None
        module = builder()
        if not isinstance(module, dspy.Module):
            raise TypeError(
                f"Builder for {name!r} returned {type(module).__name__}, expected a "
                f"dspy.Module."
            )
        return module

    def names(self) -> list[str]:
        return sorted(self._builders)

    def __contains__(self, name: object) -> bool:
        return name in self._builders


# Process-global default registry. deploy()/register_program() populate
# this; the activity reads from it at runtime.
_DEFAULT_REGISTRY = ProgramRegistry()


def default_registry() -> ProgramRegistry:
    return _DEFAULT_REGISTRY


def register_program(name: str, source: ModuleSource) -> None:
    """Register a program builder or prototype instance in the global registry."""
    _DEFAULT_REGISTRY.register(name, source)
