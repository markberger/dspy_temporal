"""Shared test fixtures."""

from types import SimpleNamespace

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM

import dspy_temporal as dt
from dspy_temporal import config as config_mod
from dspy_temporal.registry import (
    ProgramRegistry,
    default_registry,
    register_program,
)


@pytest.fixture(autouse=True)
def reset_worker_lm():
    """Snapshot/restore worker LM + tracing callback + shutdown around each test.

    The fine-mode LM-map cache eviction is NOT done here: it lives in
    ``restore_registry``'s teardown, co-located with the registry rollback it
    depends on (a rolled-back name must drop its now-stale cached map), so it
    doesn't hinge on LIFO autouse-fixture teardown ordering between the two.
    """
    saved_lm = config_mod.get_worker_lm()
    saved_cb = config_mod.get_tracing_callback()
    saved_shutdown = config_mod.get_tracing_shutdown()
    try:
        yield
    finally:
        config_mod._WORKER_LM = saved_lm
        config_mod._TRACING_CALLBACK = saved_cb
        config_mod._TRACING_SHUTDOWN = saved_shutdown


@pytest.fixture(autouse=True)
def restore_registry():
    """Snapshot the process-global registry and restore it after a test.

    Without this, the conflict guard (#30) would make a name registered by one
    test raise in the next test that re-registers it with a different object. We
    snapshot/restore (via :meth:`ProgramRegistry.snapshot` / ``restore``) rather
    than blanket-clear so any registrations made *within* a test (e.g. a
    ``ref.bind(impl)``) survive that test and then roll back cleanly between tests.

    The fine-mode LM-map cache is cleared in this fixture's OWN teardown, right
    after the registry restore: the cache is keyed off registry registrations, so
    the rollback and the cache eviction it implies belong together rather than
    split across ``reset_worker_lm`` (avoiding any dependence on autouse teardown
    order). ``_listeners`` and the generation map are deliberately NOT
    snapshotted/restored: both are process infrastructure (the cache's eviction
    hook subscribed once at import; generations only ever advance) that must
    persist across every test, not be reset per test.
    """
    from dspy_temporal.fine.activities import clear_lm_map_cache

    reg = default_registry()
    snap = reg.snapshot()
    try:
        yield
    finally:
        reg.restore(snap)
        # The per-program LM map is process-global and keyed off the registry we
        # just rolled back; drop it so a program name reused across tests with a
        # different builder doesn't see a stale map.
        clear_lm_map_cache()


@pytest.fixture
def fresh_registry():
    """A clean ProgramRegistry instance (isolated from the process-global one)."""
    return ProgramRegistry()


@pytest.fixture
def dummy_lm():
    """A canned offline LM so tests need no network or API keys."""
    return DummyLM(
        [{"reasoning": "the sky scatters blue light", "answer": "blue"}] * 50
    )


@pytest.fixture
def qa_program(dummy_lm):
    """Register a 'qa' program and set the worker LM to the dummy LM."""
    register_program(
        "qa", lambda: dspy.ChainOfThought("question -> answer"), mode=dt.RunMode.COARSE
    )
    dt.set_worker_lm(dummy_lm)
    return "qa"


# --- Fine-mode ReAct scaffolding --------------------------------------------
# Each fine-mode LM call runs in its own activity on an isolated worker_lm.copy(),
# which deep-copies instance state away. So a positional DummyLM would reset to
# its first answer every call, and per-instance counters wouldn't survive the
# copy. We therefore (a) drive responses from the *prompt* (stateless, like a
# real model) and (b) keep counts/flags in module-global dicts.

_REACT_CALLS = {"react": 0, "extract": 0, "tool": 0}
_REACT_STATE = {"extract_failed": False}


def _weather_tool(city: str) -> str:
    """Return a weather report for a city (and count the call)."""
    _REACT_CALLS["tool"] += 1
    return f"The weather in {city} is sunny."


def _build_weather_react() -> dspy.Module:
    return dspy.ReAct("question -> answer", tools=[_weather_tool])


class ReActWorkerLM(DummyLM):
    """A stateless, prompt-driven worker LM for fine-mode ReAct tests.

    Decides its response from the prompt: emit a ``get_weather`` tool call until
    the trajectory shows an observation, then ``finish``, then answer from the
    observation. With ``fail_extract_once`` the first extract-step call raises
    once (then succeeds) -- used to prove a late failure doesn't re-run the
    already-completed LM/tool activities.
    """

    def __init__(self, *, fail_extract_once: bool = False):
        super().__init__([])
        self._fail_extract_once = fail_extract_once

    def forward(self, prompt=None, messages=None, **kwargs):
        text = " ".join((m.get("content") or "") for m in (messages or []))
        if "next_tool_name" in text:  # a ReAct reasoning step
            _REACT_CALLS["react"] += 1
            if "observation_0" in text:
                fields = {
                    "next_thought": "I have the weather; finishing.",
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                }
            else:
                fields = {
                    "next_thought": "I should check the weather.",
                    "next_tool_name": "_weather_tool",
                    "next_tool_args": {"city": "Tokyo"},
                }
        else:  # the final extract step
            _REACT_CALLS["extract"] += 1
            if self._fail_extract_once and not _REACT_STATE["extract_failed"]:
                _REACT_STATE["extract_failed"] = True
                raise RuntimeError("transient extract-step failure")
            fields = {
                "reasoning": "Based on the observation.",
                "answer": "It is sunny in Tokyo.",
            }

        content = self._format_answer_fields(fields)
        message = dotdict(content=content, tool_calls=None)
        return dotdict(
            choices=[dotdict(message=message, finish_reason="stop")],
            usage=dotdict(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            model="dummy",
        )


@pytest.fixture
def fine_react():
    """Register the fine-mode 'weather_agent' ReAct program + reset counters.

    Returns a namespace with the program ``name``, the ``counters`` dict, and the
    ``worker_lm_cls`` so a test can install a failing variant. The worker LM is
    pre-set to a non-failing instance; a test may override it.
    """
    _REACT_CALLS.update(react=0, extract=0, tool=0)
    _REACT_STATE.update(extract_failed=False)
    register_program("weather_agent", _build_weather_react, mode=dt.RunMode.FINE)
    dt.set_worker_lm(ReActWorkerLM())
    return SimpleNamespace(
        name="weather_agent",
        counters=_REACT_CALLS,
        worker_lm_cls=ReActWorkerLM,
    )
