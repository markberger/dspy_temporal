"""Deploy a live (possibly compiled) ``dspy.Module`` instance.

``dt.deploy`` accepts a ``dspy.Module`` *instance* -- e.g. a program you optimized
with a DSPy teleprompter, whose predictors carry few-shot demos -- as well as a
zero-arg builder (e.g. ``lambda: dspy.ChainOfThought("question -> answer")``). The
instance stays in worker memory as a prototype; each run gets a fresh, LM-stripped
``deepcopy`` (demos preserved, no bound LM or API key ever serialized into Temporal
history).

Imported by ``examples/worker.py`` so the program registers at worker startup.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"

# A live instance. Here it is trivial, but in practice this could be the compiled
# program an optimizer returns, with demos attached to its predictors -- deploy()
# preserves those demos while stripping any bound LM.
program = dspy.ChainOfThought("question -> answer")

qa_instance = dt.deploy(
    program,
    name="qa_instance",
    mode=dt.RunMode.COARSE,
    task_queue=TASK_QUEUE,
)
