"""Tests for worker-side LM configuration (offline; no network)."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from dspy_temporal import config as cfg


def test_configure_lm_from_explicit_model_sets_worker_lm():
    lm = cfg.configure_lm_from_env(model="openai/gpt-4o-mini")
    assert isinstance(lm, dspy.LM)
    assert lm.model == "openai/gpt-4o-mini"
    assert cfg.get_worker_lm() is lm
    assert dspy.settings.lm is lm  # set_worker_lm also configures the global


def test_configure_lm_reads_model_from_env(monkeypatch):
    monkeypatch.setenv("DSPY_LM_MODEL", "openai/gpt-4o")
    lm = cfg.configure_lm_from_env()
    assert lm.model == "openai/gpt-4o"


def test_configure_lm_forwards_kwargs():
    lm = cfg.configure_lm_from_env(model="openai/gpt-4o-mini", temperature=0.0, max_tokens=128)
    assert lm.kwargs.get("temperature") == 0.0
    assert lm.kwargs.get("max_tokens") == 128


def test_configure_lm_raises_without_model(monkeypatch):
    monkeypatch.delenv("DSPY_LM_MODEL", raising=False)
    with pytest.raises(ValueError, match="No LM model configured"):
        cfg.configure_lm_from_env()


def test_worker_lm_set_get_clear():
    lm = DummyLM([{"answer": "x"}])
    cfg.set_worker_lm(lm)
    assert cfg.get_worker_lm() is lm
    cfg.clear_worker_lm()
    assert cfg.get_worker_lm() is None
