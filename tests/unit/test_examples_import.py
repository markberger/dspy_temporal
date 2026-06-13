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
