"""Host-side registration for the compose example.

Kept SEPARATE from ``compose_program.py`` (the workflow file) on purpose:
Temporal's sandbox re-execs the workflow file on every workflow task for
deterministic isolation, which would re-run any import-time ``deploy`` placed
there (rebuilding a fresh program and re-registering it each task). The
side-effecting ``deploy`` that registers ``"compose_qa"`` therefore lives here
instead; the workflow file passthrough-imports the ``triage_agent`` handle below,
so ``deploy`` runs exactly once in the worker process and never inside the
sandbox. See ``examples/compose_program.py``.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"

# A deployed program, composed into ResearchWorkflow via agent.run().
triage_agent = dt.deploy(
    lambda: dspy.ChainOfThought("question -> answer"),
    name="compose_qa",
    mode=dt.RunMode.COARSE,
    task_queue=TASK_QUEUE,
)
