"""Bind a live (possibly compiled) ``dspy.Module`` instance to a program ref.

``ref.bind(impl)`` accepts a ``dspy.Module`` *instance* -- e.g. a program you
optimized with a DSPy teleprompter, whose predictors carry few-shot demos -- as
well as a zero-arg builder (e.g. ``lambda: dspy.ChainOfThought("question ->
answer")``). The instance stays in worker memory as a prototype; each run gets a
fresh, LM-stripped ``deepcopy`` (demos preserved, no bound LM or API key ever
serialized into Temporal history).

The worker binds the prototype at startup (``qa_instance.bind(prototype)``, see
``examples/worker.py``).
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"

# A live instance. Here it is trivial, but in practice this could be the compiled
# program an optimizer returns, with demos attached to its predictors -- bind()
# preserves those demos while stripping any bound LM on each build.
prototype = dspy.ChainOfThought("question -> answer")

qa_instance = dt.program("qa_instance", mode=dt.RunMode.COARSE)
