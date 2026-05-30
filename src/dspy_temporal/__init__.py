"""dspy-temporal: deploy DSPy programs on Temporal as durable workflows."""

from __future__ import annotations

__version__ = "0.1.0"

from .client import run_program
from .coarse.activities import run_program_activity
from .coarse.api import DeployedProgram, deploy_module
from .coarse.workflow import DSPyProgramWorkflow
from .config import (
    CallOptions,
    RunConfig,
    clear_worker_lm,
    configure_lm_from_env,
    get_worker_lm,
    set_worker_lm,
)
from .converter import connect, data_converter
from .models import ProgramCallInput, ProgramCallOutput
from .registry import ProgramRegistry, default_registry, register_program
from .worker import build_worker

__all__ = [
    "__version__",
    # auto-wrap API
    "deploy_module",
    "DeployedProgram",
    "register_program",
    "run_program",
    # worker / client
    "build_worker",
    "connect",
    "data_converter",
    "configure_lm_from_env",
    "set_worker_lm",
    "get_worker_lm",
    "clear_worker_lm",
    # config / models
    "RunConfig",
    "CallOptions",
    "ProgramCallInput",
    "ProgramCallOutput",
    "ProgramRegistry",
    "default_registry",
    # workflow / activity (for custom worker wiring)
    "DSPyProgramWorkflow",
    "run_program_activity",
]
