"""Tests for ProgramRegistry validation and lookup."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

import dspy_temporal as dt
from dspy_temporal import registry as registry_mod
from dspy_temporal.registry import (
    all_named_predictors,
    default_registry,
    register_program,
    unregister_program,
)


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


# --- #30: conflict semantics + unregister ------------------------------------


def test_register_same_object_twice_is_noop(fresh_registry):
    """Re-registering the SAME object under a name is a no-op (no raise), e.g. a
    worker re-importing a module that registers at import time."""

    def builder():
        return dspy.Predict("q -> a")

    fresh_registry.register("x", builder)
    fresh_registry.register("x", builder)  # same object -> silently ignored
    assert fresh_registry.names() == ["x"]


def test_register_conflicting_object_raises(fresh_registry):
    """A DIFFERENT object under an already-taken name raises, and the message
    points at unregister() and lists the registered names."""
    fresh_registry.register("x", lambda: dspy.Predict("q -> a"))
    with pytest.raises(ValueError, match=r"already registered.*unregister"):
        fresh_registry.register("x", lambda: dspy.Predict("q -> a"))


def test_conflict_message_lists_multiple_registered_names(fresh_registry):
    """The conflict error enumerates every currently-registered name (sorted)."""
    fresh_registry.register("a", lambda: dspy.Predict("q -> a"))
    fresh_registry.register("b", lambda: dspy.Predict("q -> a"))
    with pytest.raises(ValueError, match=r"Registered: \['a', 'b'\]"):
        fresh_registry.register("a", lambda: dspy.Predict("q -> a"))


def test_unregister_then_register_different_object_succeeds(fresh_registry):
    """unregister() frees the name so a deliberate replacement registers cleanly."""
    fresh_registry.register("x", lambda: dspy.Predict("q -> a"))
    fresh_registry.unregister("x")
    assert "x" not in fresh_registry
    # A different object now registers without conflict.
    fresh_registry.register("x", lambda: dspy.ChainOfThought("q -> a"))
    assert "x" in fresh_registry


def test_unregister_unknown_name_is_noop(fresh_registry):
    """Unregistering a name that was never registered does not raise."""
    fresh_registry.unregister("never_here")  # silent no-op


def test_bad_source_does_not_poison_sources(fresh_registry):
    """A non-module/non-callable raises TypeError WITHOUT recording the name, so a
    later VALID register of the same name succeeds (the conflict guard isn't
    tripped by a half-written registration)."""
    with pytest.raises(TypeError, match=r"OR a dspy\.Module instance"):
        fresh_registry.register("x", object())
    assert "x" not in fresh_registry
    # The failed attempt left no trace, so a real builder registers fine.
    fresh_registry.register("x", lambda: dspy.Predict("q -> a"))
    assert "x" in fresh_registry


def test_unregister_program_module_wrapper(dummy_lm):
    """The module-level unregister_program removes a name from the global registry
    (the restore_registry fixture rolls back any leftover)."""
    register_program("temp_prog", lambda: dspy.Predict("q -> a"))
    assert "temp_prog" in default_registry()
    unregister_program("temp_prog")
    assert "temp_prog" not in default_registry()


# --- #36: snapshot/restore isolation (paired cross-test leak checks) ----------
# These two tests register the SAME global name with DIFFERENT objects. If the
# autouse restore_registry fixture didn't roll the registry back between them,
# the second would hit #30's conflict raise (or see the other's builder). Each
# asserts its own registration is present and the other's hasn't leaked in.


def test_snapshot_restore_no_leak_first():
    register_program("leaky", lambda: dspy.Predict("first -> out"))
    assert "leaky" in default_registry()


def test_snapshot_restore_no_leak_second():
    # If "leaky" leaked from the first test (same name, different object), this
    # register_program would raise #30's conflict error -- so a clean pass proves
    # the snapshot/restore rolled the first test's registration back.
    register_program("leaky", lambda: dspy.Predict("second -> out"))
    assert "leaky" in default_registry()


def test_snapshot_restore_modes_do_not_leak_set():
    """Set a mode on a name; a sibling test asserts mode_for is None for it,
    proving _modes is snapshot/restored too (not just _builders)."""
    register_program("moded", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.FINE)
    assert default_registry().mode_for("moded") == dt.RunMode.FINE


def test_snapshot_restore_modes_do_not_leak_check():
    # "moded" registered with FINE in the sibling test must not survive here.
    assert default_registry().mode_for("moded") is None


# --- #29: mode storage + mode_for --------------------------------------------


def test_register_program_stores_mode_and_mode_for_reads_it(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.FINE)
    assert fresh_registry.mode_for("m") == dt.RunMode.FINE


def test_mode_for_none_when_registered_without_mode(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"))
    assert fresh_registry.mode_for("m") is None


def test_mode_for_none_when_unregistered(fresh_registry):
    assert fresh_registry.mode_for("absent") is None


def test_unregister_drops_mode(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.COARSE)
    fresh_registry.unregister("m")
    assert fresh_registry.mode_for("m") is None


# --- #28: invalidation listener hook -----------------------------------------


def test_add_invalidation_listener_fires_on_register(fresh_registry):
    fired = []
    fresh_registry.add_invalidation_listener(fired.append)
    fresh_registry.register("a", lambda: dspy.Predict("q -> a"))
    assert fired == ["a"]


def test_invalidation_listener_fires_again_after_unregister_register(fresh_registry):
    """A genuine replace fires the listener on EACH step: the register, the
    unregister (now also an invalidation), and the re-register -- every one of
    which the fine cache must observe to drop the program's stale LM map."""
    fired = []
    fresh_registry.add_invalidation_listener(fired.append)
    fresh_registry.register("a", lambda: dspy.Predict("q -> a"))
    fresh_registry.unregister("a")
    fresh_registry.register("a", lambda: dspy.ChainOfThought("q -> a"))
    assert fired == ["a", "a", "a"]


def test_invalidation_listener_not_fired_on_same_object_noop(fresh_registry):
    """The same-object re-register returns early, BEFORE the listener loop, so it
    does not fire (a no-op must not invalidate caches)."""
    fired = []

    def builder():
        return dspy.Predict("q -> a")

    fresh_registry.add_invalidation_listener(fired.append)
    fresh_registry.register("a", builder)
    fresh_registry.register("a", builder)  # same object -> no-op, no fire
    assert fired == ["a"]


def test_unregister_fires_invalidation_listener(fresh_registry):
    """Unregistering a present name fires the listener (and bumps the generation),
    so a cache evicts the entry that's now gone -- not only re-registration does."""
    fired = []
    fresh_registry.add_invalidation_listener(fired.append)
    fresh_registry.register("a", lambda: dspy.Predict("q -> a"))
    gen_after_register = fresh_registry.generation("a")
    fresh_registry.unregister("a")
    assert fired == ["a", "a"]  # once on register, once on unregister
    assert fresh_registry.generation("a") == gen_after_register + 1


def test_unregister_unknown_name_does_not_fire(fresh_registry):
    """Unregistering a name that was never registered fires nothing and leaves the
    generation untouched, so listener-fired counts stay precise."""
    fired = []
    fresh_registry.add_invalidation_listener(fired.append)
    fresh_registry.unregister("never_here")
    assert fired == []
    assert fresh_registry.generation("never_here") == 0


# --- snapshot / restore -------------------------------------------------------


def test_snapshot_restore_roundtrip(fresh_registry):
    """snapshot() then restore() rolls the entry map (builders, sources, AND modes)
    back to the captured state -- the primitive the autouse fixture relies on."""
    fresh_registry.register(
        "keep", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.FINE
    )
    snap = fresh_registry.snapshot()

    # Mutate after the snapshot: add a name, drop the original, replace its mode.
    fresh_registry.register("added", lambda: dspy.Predict("q -> a"))
    fresh_registry.unregister("keep")
    assert "added" in fresh_registry
    assert "keep" not in fresh_registry

    fresh_registry.restore(snap)
    # The added name is gone, the dropped name is back, and its mode is restored.
    assert fresh_registry.names() == ["keep"]
    assert "added" not in fresh_registry
    assert fresh_registry.mode_for("keep") == dt.RunMode.FINE
    assert isinstance(fresh_registry.build("keep"), dspy.Module)


# --- resolve_mode: the by-name run-mode resolution ladder ---------------------
# The client's run_program delegates here; test every branch + exact raise text.


def test_resolve_mode_registered_with_mode_omitted_returns_registered(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.FINE)
    assert fresh_registry.resolve_mode("m", None) == dt.RunMode.FINE


def test_resolve_mode_registered_with_mode_explicit_equal_ok(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.COARSE)
    assert fresh_registry.resolve_mode("m", dt.RunMode.COARSE) == dt.RunMode.COARSE


def test_resolve_mode_registered_with_mode_explicit_conflict_raises(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"), mode=dt.RunMode.COARSE)
    with pytest.raises(
        ValueError, match=r"registered as mode='coarse'.*mode='fine'.*handle\.start"
    ):
        fresh_registry.resolve_mode("m", dt.RunMode.FINE)


def test_resolve_mode_registered_without_mode_omitted_raises(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"))  # no mode
    with pytest.raises(ValueError, match=r"registered without a run mode"):
        fresh_registry.resolve_mode("m", None)


def test_resolve_mode_registered_without_mode_explicit_is_returned(fresh_registry):
    fresh_registry.register("m", lambda: dspy.Predict("q -> a"))  # no mode
    assert fresh_registry.resolve_mode("m", dt.RunMode.FINE) == dt.RunMode.FINE


def test_resolve_mode_unregistered_omitted_raises(fresh_registry):
    with pytest.raises(ValueError, match=r"not registered in this process"):
        fresh_registry.resolve_mode("absent", None)


def test_resolve_mode_unregistered_explicit_is_returned(fresh_registry):
    assert fresh_registry.resolve_mode("absent", dt.RunMode.COARSE) == dt.RunMode.COARSE


# --- sandbox guardrail on the module-level register_program -------------------


def test_register_program_refused_in_sandbox(monkeypatch):
    """The module-level register_program refuses to run inside the Temporal
    workflow sandbox (a top-level ref.bind() in a workflow file re-execs each task).
    Uses workflow.unsafe.in_sandbox() -- in_workflow() is False during re-exec."""
    monkeypatch.setattr(
        registry_mod.workflow.unsafe, "in_sandbox", lambda: True, raising=True
    )
    with pytest.raises(RuntimeError, match=r"sandbox.*compose_refs\.py"):
        register_program("in_sandbox", lambda: dspy.Predict("q -> a"))
    # The refusal happened before any mutation -- the name never registered.
    assert "in_sandbox" not in default_registry()


def test_register_program_allowed_off_sandbox_thread():
    """Off the sandbox thread (the normal host import path) in_sandbox() is False,
    so register_program proceeds -- the guard must not false-positive."""
    register_program("host_ok", lambda: dspy.Predict("q -> a"))
    assert "host_ok" in default_registry()
