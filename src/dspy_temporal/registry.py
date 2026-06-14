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


def _copy_stripped(module: dspy.Module) -> dspy.Module:
    """Return a fresh copy of ``module`` with every predictor's ``.lm`` dropped.

    ``deepcopy`` (not ``reset_copy``) so a compiled program's few-shot demos are
    preserved; nulling each predictor's ``.lm`` drops any bound live LM (and its
    API keys) so the prototype's secrets never reach a built copy. Pure-Python and
    I/O-free, so it is safe to call from the fine workflow's sandboxed build path.
    """
    clone = module.deepcopy()
    for _name, predictor in clone.named_predictors():
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
