"""Side-effect-free program reference for the compose example.

``dt.program(...)`` is a pure declaration: no registry mutation, no model load, no
dspy at the source level. That makes this module safe to import *normally* from
both the workflow file (``compose_program.py``) and a thin client -- no
``imports_passed_through()`` dance, no import-time ``deploy`` to re-run inside the
sandbox, and the workflow class stays cheap to import (so a client can start it
type-safely with ``start_workflow(ResearchWorkflow.run, ...)``).

The heavy implementation is attached on the worker via
``triage_agent.bind(...)`` -- see ``examples/worker.py``.
"""

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"

# A program reference, composed into ResearchWorkflow via ``triage_agent.run()``.
triage_agent = dt.program("compose_qa", mode=dt.RunMode.COARSE)
