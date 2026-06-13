"""Shared test fixtures."""

from types import SimpleNamespace

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM

import dspy_temporal as dt
from dspy_temporal import config as config_mod
from dspy_temporal.registry import ProgramRegistry


@pytest.fixture(autouse=True)
def reset_worker_lm():
    """Snapshot/restore worker LM + tracing callback around each test."""
    saved_lm = config_mod.get_worker_lm()
    saved_cb = config_mod.get_tracing_callback()
    try:
        yield
    finally:
        config_mod._WORKER_LM = saved_lm
        config_mod._TRACING_CALLBACK = saved_cb


@pytest.fixture
def fresh_registry():
    """A clean ProgramRegistry instance (isolated from the process-global one)."""
    return ProgramRegistry()


@pytest.fixture
def dummy_lm():
    """A canned offline LM so tests need no network or API keys."""
    return DummyLM([{"reasoning": "the sky scatters blue light", "answer": "blue"}] * 50)


@pytest.fixture
def qa_program(dummy_lm):
    """Register a 'qa' program and set the worker LM to the dummy LM."""
    dt.register_program("qa", lambda: dspy.ChainOfThought("question -> answer"))
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
            fields = {"reasoning": "Based on the observation.", "answer": "It is sunny in Tokyo."}

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
    dt.register_program("weather_agent", _build_weather_react)
    dt.set_worker_lm(ReActWorkerLM())
    return SimpleNamespace(
        name="weather_agent",
        counters=_REACT_CALLS,
        worker_lm_cls=ReActWorkerLM,
    )
