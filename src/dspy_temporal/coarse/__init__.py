"""Coarse execution mode: run a whole dspy.Module inside one Temporal activity."""

from .activities import run_program_activity
from .workflow import DSPyProgramWorkflow

__all__ = ["run_program_activity", "DSPyProgramWorkflow"]
