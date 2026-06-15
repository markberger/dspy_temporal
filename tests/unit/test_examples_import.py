"""Smoke test: the documented examples declare pure refs and bind cleanly.

This guards the onboarding path shown in the README without needing a server.
``dt.program(...)`` is pure -- importing an example registers nothing -- so these
tests assert the declaration is side-effect-free and that the worker's explicit
``ref.bind(impl)`` step registers it.
"""

import importlib
import sys
from pathlib import Path

import dspy

from dspy_temporal.registry import default_registry

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _import_example(module_name):
    """Import an example module with the examples dir on the path."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(EXAMPLES_DIR))


def test_example_qa_program_is_pure_then_binds():
    qa_program = _import_example("qa_program")

    # program() is pure: importing the declaration registers nothing.
    assert "qa" not in default_registry()
    assert qa_program.qa.name == "qa"

    # The worker's explicit bind step is what registers the implementation.
    qa_program.qa.bind(qa_program.build_qa)
    assert "qa" in default_registry()


def test_example_instance_program_binds_live_instance():
    """ref.bind() accepts a live dspy.Module instance (a compiled prototype)."""
    instance_program = _import_example("instance_program")

    assert "qa_instance" not in default_registry()
    instance_program.qa_instance.bind(instance_program.prototype)
    assert "qa_instance" in default_registry()
    # The registered prototype builds an LM-stripped copy.
    built = default_registry().build("qa_instance")
    assert all(p.lm is None for _n, p in built.named_predictors())


def test_example_compose_program_declares_workflow_and_ref():
    """A user @workflow.defn composing triage_agent.run(), with a pure ref."""
    compose_program = _import_example("compose_program")

    # The ref is declared side-effect-free; binding is the worker's job.
    assert "compose_qa" not in default_registry()
    assert compose_program.triage_agent.name == "compose_qa"
    # The composed workflow is a real @workflow.defn (has a run method).
    assert hasattr(compose_program.ResearchWorkflow, "run")

    compose_program.triage_agent.bind(lambda: dspy.ChainOfThought("question -> answer"))
    assert "compose_qa" in default_registry()
