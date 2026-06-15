"""Side-effect-free program reference for the compose example.

``dt.program(...)`` is a pure declaration: no registry mutation, no model load, no
dspy at the source level. That makes this module safe to import *normally* from
both the workflow file (``compose_program.py``) and a thin client -- no
``imports_passed_through()`` dance, no import-time ``bind`` to re-run inside the
sandbox, and the workflow class stays cheap to import (so a client can start it
type-safely with ``start_workflow(ResearchWorkflow.run, ...)``).

The reference also declares a ``result`` adapter, so ``triage_agent.run(...)``
returns a typed ``Answer`` (a pydantic model) rather than a raw ``dspy.Prediction``
-- dspy never leaks into the workflow or the caller.

The heavy implementation is attached on the worker via
``triage_agent.bind(...)`` -- see ``examples/worker.py``.
"""

from pydantic import BaseModel

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"


class Answer(BaseModel):
    """The typed result the compose example returns instead of a raw Prediction."""

    text: str


# A program reference, composed into ResearchWorkflow via ``triage_agent.run()``.
# The ``result`` adapter shapes each run's ``dspy.Prediction`` into an ``Answer``,
# so workflow and caller code speak pydantic, not dspy.
triage_agent = dt.program(
    "compose_qa",
    mode=dt.RunMode.COARSE,
    result=lambda p: Answer(text=str(p.answer)),
)
