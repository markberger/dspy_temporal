"""Shared test fixtures."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

import dspy_temporal as dt


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
