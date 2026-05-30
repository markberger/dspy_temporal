"""Program registry: maps a program *name* to a zero-arg builder.

Workflows and activity inputs carry only a program name plus call inputs --
never a live ``dspy.Module`` (which would serialize child predictors' LM
objects, including API keys, into durable Temporal history). The worker process
owns the registry and reconstructs a fresh module on demand via its builder.
"""

from __future__ import annotations

from typing import Callable

import dspy

ModuleBuilder = Callable[[], dspy.Module]


class ProgramRegistry:
    """A name -> builder mapping owned by the worker process."""

    def __init__(self) -> None:
        self._builders: dict[str, ModuleBuilder] = {}

    def register(self, name: str, builder: ModuleBuilder) -> None:
        if not callable(builder):
            raise TypeError(
                f"Program {name!r} builder must be a zero-arg callable returning a "
                f"dspy.Module, got {type(builder).__name__}."
            )
        self._builders[name] = builder

    def build(self, name: str) -> dspy.Module:
        try:
            builder = self._builders[name]
        except KeyError:
            raise KeyError(
                f"No program registered under {name!r}. Registered: "
                f"{sorted(self._builders)}. Did the worker import the module that "
                f"calls deploy_module()/register_program()?"
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


# Process-global default registry. deploy_module()/register_program() populate
# this; the activity reads from it at runtime.
_DEFAULT_REGISTRY = ProgramRegistry()


def default_registry() -> ProgramRegistry:
    return _DEFAULT_REGISTRY


def register_program(name: str, builder: ModuleBuilder) -> None:
    """Register a program builder in the process-global registry."""
    _DEFAULT_REGISTRY.register(name, builder)
