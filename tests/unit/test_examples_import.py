"""Smoke test: the documented example imports and registers its program.

This guards the onboarding path shown in the README without needing a server.
"""

import importlib
import sys
from pathlib import Path

from dspy_temporal.registry import default_registry

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _import_example(module_name, *also_evict):
    """Import an example module, forcing a *fresh* execution of its body.

    Examples ``deploy``/``register_program`` at import time. Python caches a
    module after its first import, so a plain ``import`` re-runs that body only
    once per session -- but the autouse ``restore_registry`` fixture rolls each
    test's registry back afterward. We evict the module first so its import-time
    registration re-runs inside *this* test's snapshot window (and is asserted
    present), then rolled back like any other. ``also_evict`` names any
    side-effecting dependency module that must likewise re-run (e.g. a workflow
    file's separate registration module).
    """
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        for mod in (module_name, *also_evict):
            sys.modules.pop(mod, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(EXAMPLES_DIR))


def test_example_qa_program_registers():
    qa_program = _import_example("qa_program")

    assert "qa" in default_registry()
    assert qa_program.qa.name == "qa"
    assert qa_program.qa.task_queue == qa_program.TASK_QUEUE


def test_example_deploy_instance_registers():
    """deploy() wraps a live dspy.Module instance."""
    deploy_instance = _import_example("deploy_instance")

    assert "qa_instance" in default_registry()
    assert deploy_instance.qa_instance.name == "qa_instance"
    # The registered prototype builds an LM-stripped copy.
    built = default_registry().build("qa_instance")
    assert all(p.lm is None for _n, p in built.named_predictors())


def test_example_compose_program_registers():
    """A user @workflow.defn composing agent.run()."""
    # compose_program passthrough-imports its deploy from compose_agents; evict
    # both so the registration re-runs in this test's snapshot window.
    compose_program = _import_example("compose_program", "compose_agents")

    assert "compose_qa" in default_registry()
    assert compose_program.triage_agent.name == "compose_qa"
    # The composed workflow is a real @workflow.defn (has a run method).
    assert hasattr(compose_program.ResearchWorkflow, "run")
