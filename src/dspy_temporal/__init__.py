"""dspy-temporal: deploy DSPy programs on Temporal as durable workflows."""

from __future__ import annotations

__version__ = "0.1.0"

from .client import run_program
from .coarse.api import TemporalProgram, deploy
from .config import (
    CallOptions,
    RunMode,
    configure_lm_from_env,
    set_worker_lm,
)
from .converter import connect
from .plugin import DSPyPlugin
from .worker import build_worker

# Strict surface: ``dt.X`` exists **iff** ``X`` is documented in ``__all__``.
# Everything else lives in its defining submodule and is imported from there:
#   - wire models -> ``dspy_temporal.models``
#   - raw workflow/activity classes, ``DSPY_ACTIVITIES`` / ``DSPY_WORKFLOWS`` ->
#     ``dspy_temporal.plugin`` (and ``.coarse``/``.fine``)
#   - the in-your-workflow compose seams ``execute_coarse`` / ``execute_fine`` ->
#     ``dspy_temporal.execute`` (``agent.run()`` is the headline compose verb)
#   - ``data_converter`` -> ``dspy_temporal.converter``
#   - ``register_program`` / ``default_registry`` / ``ProgramRegistry`` ->
#     ``dspy_temporal.registry``
#   - ``get_worker_lm`` / ``clear_worker_lm`` -> ``dspy_temporal.config``
__all__ = [  # noqa: RUF022 -- grouped by concern, not alphabetized
    "__version__",
    # deploy + run a program
    "deploy",  # deploy(source, *, name, task_queue, mode=COARSE)
    "TemporalProgram",  # handle type returned by deploy (run / start)
    "run_program",  # low-level by-name standalone start (task_queue required)
    # worker / client wiring
    "build_worker",  # build_worker(client, *, task_queue, ...)
    "DSPyPlugin",
    "connect",
    "configure_lm_from_env",  # LM from env (the documented LM setup)
    "set_worker_lm",  # bring-your-own dspy.LM object (advanced LM setup)
    # config / tuning
    "RunMode",
    "CallOptions",  # retry/timeout tuning for a run
]
