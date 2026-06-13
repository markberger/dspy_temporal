"""dspy-temporal: deploy DSPy programs on Temporal as durable workflows."""

from __future__ import annotations

__version__ = "0.1.0"

from .client import run_program
from .coarse.activities import run_program_activity
from .coarse.api import DeployedProgram, TemporalProgram, deploy, deploy_module
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

__all__ = [  # noqa: RUF022 -- grouped by concern with section comments, not alphabetized
    "__version__",
    # auto-wrap API
    "deploy",
    "deploy_module",
    "TemporalProgram",
    "DeployedProgram",
    "register_program",
    "run_program",
    # compose-in-your-own-workflow (Win B)
    "execute_coarse",
    "execute_fine",
    # worker / client
    "build_worker",
    "DSPyPlugin",
    "DSPY_ACTIVITIES",
    "DSPY_WORKFLOWS",
    "connect",
    "data_converter",
    "configure_lm_from_env",
    "set_worker_lm",
    "get_worker_lm",
    "clear_worker_lm",
    # config / models
    "RunConfig",
    "RunMode",
    "CallOptions",
    "ProgramCallInput",
    "ProgramCallOutput",
    "LMCallInput",
    "LMCallOutput",
    "LMSpec",
    "LMDescribeInput",
    "LMSpecsOutput",
    "ToolCallInput",
    "ToolCallOutput",
    "ProgramRegistry",
    "default_registry",
    # workflow / activity (for custom worker wiring)
    "DSPyProgramWorkflow",
    "run_program_activity",
    # fine mode
    "DSPyProgramFineWorkflow",
    "describe_lms_activity",
    "lm_call_activity",
    "tool_call_activity",
]
