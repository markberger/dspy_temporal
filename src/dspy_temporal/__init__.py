"""dspy-temporal: deploy DSPy programs on Temporal as durable workflows."""

from __future__ import annotations

__version__ = "0.1.0"

from .client import run_program
from .coarse.activities import run_program_activity
from .coarse.api import TemporalProgram, deploy
from .coarse.workflow import DSPyProgramWorkflow
from .config import (
    CallOptions,
    RunConfig,
    RunMode,
    clear_worker_lm,
    configure_lm_from_env,
    get_worker_lm,
    set_worker_lm,
)
from .converter import connect, data_converter
from .execute import execute_coarse, execute_fine
from .fine.activities import (
    describe_lms_activity,
    lm_call_activity,
    tool_call_activity,
)
from .fine.workflow import DSPyProgramFineWorkflow
from .models import (
    LMCallInput,
    LMCallOutput,
    LMDescribeInput,
    LMSpec,
    LMSpecsOutput,
    ProgramCallInput,
    ProgramCallOutput,
    ToolCallInput,
    ToolCallOutput,
)
from .plugin import DSPY_ACTIVITIES, DSPY_WORKFLOWS, DSPyPlugin
from .registry import ProgramRegistry, default_registry, register_program
from .worker import build_worker

# ``__all__`` is the *headline* surface: the verbs a user needs for the common
# path (deploy a program, run it, wire a worker, connect, pick a mode). Everything
# imported above stays importable as ``dt.<name>`` -- the registry (``register_program``,
# ``ProgramRegistry``, ``default_registry``), the wire models (``ProgramCallInput`` &
# co.), the raw workflow/activity classes (``DSPY_ACTIVITIES`` / ``DSPY_WORKFLOWS``,
# ``DSPyProgramWorkflow`` / ``run_program_activity`` / the fine-mode set), the
# worker-LM setters (``set_worker_lm`` / ``get_worker_lm`` / ``clear_worker_lm``),
# ``data_converter`` and ``CallOptions`` -- they are just kept out of the advertised
# list to keep the core legible. (``__init__.py`` is F401-exempt so these re-exports
# don't trip "imported but unused".)
__all__ = [  # noqa: RUF022 -- grouped by concern, not alphabetized
    "__version__",
    # deploy + run a program
    "deploy",
    "run_program",
    "TemporalProgram",
    # worker / client
    "build_worker",
    "DSPyPlugin",
    "connect",
    "configure_lm_from_env",
    # config
    "RunConfig",
    "RunMode",
    # compose a deployed program into your own workflow
    "execute_coarse",
    "execute_fine",
]
