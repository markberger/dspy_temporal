"""Smoke test: the documented example imports and registers its program.

This guards the onboarding path shown in the README without needing a server.
"""

import sys
from pathlib import Path

import dspy_temporal as dt

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_example_qa_program_registers():
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        import qa_program
    finally:
        sys.path.remove(str(EXAMPLES_DIR))

    assert "qa" in dt.default_registry()
    assert qa_program.qa.name == "qa"
    assert qa_program.qa.config.task_queue == qa_program.TASK_QUEUE


def test_example_deploy_instance_registers():
    """deploy() wraps a live dspy.Module instance."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        import deploy_instance
    finally:
        sys.path.remove(str(EXAMPLES_DIR))

    assert "qa_instance" in dt.default_registry()
    assert deploy_instance.qa_instance.name == "qa_instance"
    # The registered prototype builds an LM-stripped copy.
    built = dt.default_registry().build("qa_instance")
    assert all(p.lm is None for _n, p in built.named_predictors())


def test_example_compose_program_registers():
    """A user @workflow.defn composing agent.run()."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        import compose_program
    finally:
        sys.path.remove(str(EXAMPLES_DIR))

    assert "compose_qa" in dt.default_registry()
    assert compose_program.triage_agent.name == "compose_qa"
    # The composed workflow is a real @workflow.defn (has a run method).
    assert hasattr(compose_program.ResearchWorkflow, "run")
