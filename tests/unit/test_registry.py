"""Tests for ProgramRegistry validation and lookup."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM


def test_register_rejects_non_module_non_callable(fresh_registry):
    # object() is neither callable nor a dspy.Module -> the else TypeError branch.
    with pytest.raises(TypeError, match=r"OR a dspy\.Module instance"):
        fresh_registry.register("bad", object())


def test_build_unknown_name_raises_helpful_keyerror(fresh_registry):
    fresh_registry.register("known", lambda: dspy.Predict("q -> a"))
    with pytest.raises(KeyError, match="No program registered under 'missing'"):
        fresh_registry.build("missing")


def test_build_rejects_non_module(fresh_registry):
    fresh_registry.register("notmod", lambda: 42)
    with pytest.raises(TypeError, match=r"expected a dspy\.Module"):
        fresh_registry.build("notmod")


def test_build_returns_module(fresh_registry):
    # The elif-callable (builder) branch.
    fresh_registry.register("ok", lambda: dspy.Predict("q -> a"))
    assert isinstance(fresh_registry.build("ok"), dspy.Module)


def test_register_accepts_instance_and_builds_stripped_copy(fresh_registry):
    """An instance is cloned LM-stripped per build: demos preserved (compiled
    programs survive), bound LMs dropped (no secrets), prototype left intact."""
    prototype = dspy.ChainOfThought("question -> answer")
    for _name, predictor in prototype.named_predictors():
        predictor.lm = DummyLM([{"reasoning": "r", "answer": "a"}])
        predictor.demos = [{"question": "q", "answer": "a"}]

    fresh_registry.register("proto", prototype)
    built = fresh_registry.build("proto")

    # Distinct object, every predictor LM-stripped, demos preserved.
    assert built is not prototype
    built_predictors = dict(built.named_predictors())
    assert built_predictors
    for predictor in built_predictors.values():
        assert predictor.lm is None
        assert predictor.demos == [{"question": "q", "answer": "a"}]
    # The prototype keeps its bound LM (only the copy is stripped).
    assert all(p.lm is not None for _n, p in prototype.named_predictors())


def test_register_instance_build_returns_fresh_each_time(fresh_registry):
    """Each build() is an independent deepcopy -- no shared mutable demos/traces
    leaking across concurrent runs."""
    fresh_registry.register("proto", dspy.ChainOfThought("question -> answer"))
    first = fresh_registry.build("proto")
    second = fresh_registry.build("proto")
    assert first is not second
    first_predictor = next(iter(dict(first.named_predictors()).values()))
    second_predictor = next(iter(dict(second.named_predictors()).values()))
    assert first_predictor is not second_predictor


def test_names_sorted_and_contains(fresh_registry):
    fresh_registry.register("b", lambda: dspy.Predict("q -> a"))
    fresh_registry.register("a", lambda: dspy.Predict("q -> a"))
    assert fresh_registry.names() == ["a", "b"]
    assert "a" in fresh_registry
    assert "zzz" not in fresh_registry
