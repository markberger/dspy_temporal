"""Tests for ProgramRegistry validation and lookup."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from dspy_temporal.registry import all_named_predictors


class _Inner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")

    def forward(self, **kwargs):
        return self.qa(**kwargs)


class _Mid(dspy.Module):
    def __init__(self):
        super().__init__()
        self.inner = _Inner()

    def forward(self, **kwargs):
        return self.inner(**kwargs)


class _Outer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.mid = _Mid()
        self.top = dspy.Predict("question -> answer")

    def forward(self, **kwargs):
        return self.mid(**kwargs)


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


def test_all_named_predictors_includes_nested_compiled_and_restores_flags():
    """``all_named_predictors`` reaches predictors inside (even nested) compiled
    sub-modules, names them with the bare dotted convention (NOT the ``self.``
    prefix ``named_sub_modules`` uses), and restores every ``_compiled`` flag."""
    program = _Outer()
    program.mid._compiled = True  # compiled at one level...
    program.mid.inner._compiled = True  # ...and nested inside it

    # DSPy's own named_predictors() is blind to everything under the compiled mid.
    assert [name for name, _ in program.named_predictors()] == ["top"]

    found = all_named_predictors(program)
    assert [name for name, _ in found] == ["mid.inner.qa", "top"]

    # The temporarily-cleared flags are restored on every level after the walk.
    assert program.mid._compiled is True
    assert program.mid.inner._compiled is True


def test_register_strips_lm_inside_compiled_submodule(fresh_registry):
    """The core fix: a predictor inside a *compiled* sub-module is still
    LM-stripped on build, so its bound LM (and any API key) never survives the
    clone -- even though DSPy's ``named_predictors()`` skips compiled sub-modules
    and would otherwise leak it."""
    prototype = _Outer()
    prototype.mid.inner._compiled = True  # as an optimizer would mark it
    for predictor in (prototype.mid.inner.qa, prototype.top):
        predictor.lm = DummyLM([{"answer": "a"}])
        predictor.demos = [{"question": "q", "answer": "a"}]

    fresh_registry.register("nested", prototype)
    built = fresh_registry.build("nested")

    # Verify via named_sub_modules (which DOES descend into compiled modules);
    # named_predictors() is itself blind to the compiled predictor, so asserting
    # over it would silently pass even if the leak were unfixed.
    built_predictors = list(built.named_sub_modules(type_=dspy.Predict))
    assert {name for name, _ in built_predictors} == {"self.mid.inner.qa", "self.top"}
    for _name, predictor in built_predictors:
        assert predictor.lm is None  # bound LM (+ key) dropped, including compiled
        assert predictor.demos == [{"question": "q", "answer": "a"}]  # demos kept

    # The prototype is untouched: its LMs stay bound and it stays compiled.
    assert prototype.mid.inner._compiled is True
    assert prototype.mid.inner.qa.lm is not None
    assert prototype.top.lm is not None
