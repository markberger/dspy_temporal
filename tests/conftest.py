"""Shared test fixtures."""

import dspy
import pytest
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
