"""Smoke test: the top-level package imports, and temporalio meets the floor.

Regression guard for #25. The declared ``temporalio`` floor was once ``>=1.7``,
but the code imports ``temporalio.worker.Plugin`` / ``temporalio.client.Plugin``,
the pydantic data converter, and the sandbox passthrough API -- all post-1.7.
A resolver that picked an old ``temporalio`` would fail at ``import dspy_temporal``
with ``ImportError: cannot import name 'Plugin' from 'temporalio.worker'``.

A plain ``import dspy_temporal`` transitively imports ``plugin.py`` (which
subclasses both Plugin ABCs), ``converter.py``, and ``sandbox.py`` -- so reaching
the headline symbols below at all exercises every one of those failure modes.

Note the limit: this suite resolves a single (locked) ``temporalio``, so it can't
prove that an *old* version raises -- only that the resolved one imports cleanly
and meets the declared floor. The "old version breaks" guarantee lives in the
floor declaration in ``pyproject.toml`` itself.
"""

from importlib.metadata import version

import dspy_temporal as dt

# Keep in sync with the temporalio floor in pyproject.toml.
TEMPORALIO_FLOOR = (1, 16)


def test_package_imports_with_plugin_surface():
    # If the plugin/converter/sandbox imports broke, `import dspy_temporal`
    # above would already have raised; touching the symbols documents intent.
    assert hasattr(dt, "DSPyPlugin")
    assert hasattr(dt, "connect")


def test_strict_surface_keeps_internals_off_top_level():
    """``dt.X`` exists iff X is in ``__all__``: internals are submodule-only.

    The converter is the canary -- it used to be a top-level re-export; now it
    lives only in ``dspy_temporal.converter`` (still importable from there).
    """
    assert not hasattr(dt, "data_converter")
    from dspy_temporal.converter import data_converter  # noqa: F401

    # Every advertised name resolves, and nothing outside ``__all__`` leaks.
    for name in dt.__all__:
        assert hasattr(dt, name), name
    for internal in (
        "RunConfig",
        "execute_coarse",
        "execute_fine",
        "register_program",
        "default_registry",
        "DSPY_ACTIVITIES",
    ):
        assert not hasattr(dt, internal), internal


def test_temporalio_runtime_meets_declared_floor():
    major, minor = (int(part) for part in version("temporalio").split(".")[:2])
    assert (major, minor) >= TEMPORALIO_FLOOR
