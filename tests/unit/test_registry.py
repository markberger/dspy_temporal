"""Tests for ProgramRegistry validation and lookup."""

import dspy
import pytest

from dspy_temporal.registry import ProgramRegistry


def test_register_rejects_non_callable(fresh_registry):
    with pytest.raises(TypeError, match="zero-arg callable"):
        fresh_registry.register("bad", object())  # not callable


def test_build_unknown_name_raises_helpful_keyerror(fresh_registry):
    fresh_registry.register("known", lambda: dspy.Predict("q -> a"))
    with pytest.raises(KeyError, match="No program registered under 'missing'"):
        fresh_registry.build("missing")


def test_build_rejects_non_module(fresh_registry):
    fresh_registry.register("notmod", lambda: 42)
    with pytest.raises(TypeError, match="expected a dspy.Module"):
        fresh_registry.build("notmod")


def test_build_returns_module(fresh_registry):
    fresh_registry.register("ok", lambda: dspy.Predict("q -> a"))
    assert isinstance(fresh_registry.build("ok"), dspy.Module)


def test_names_sorted_and_contains(fresh_registry):
    fresh_registry.register("b", lambda: dspy.Predict("q -> a"))
    fresh_registry.register("a", lambda: dspy.Predict("q -> a"))
    assert fresh_registry.names() == ["a", "b"]
    assert "a" in fresh_registry
    assert "zzz" not in fresh_registry
