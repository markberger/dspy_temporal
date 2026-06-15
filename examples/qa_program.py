"""Shared program *declaration* imported by both the worker and the starter.

``dt.program(...)`` builds a pure reference -- no registry mutation, no model
load. The worker attaches the implementation at startup with ``qa.bind(build_qa)``
(see ``examples/worker.py``); the starter only needs the reference's name + mode.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"


def build_qa() -> dspy.Module:
    # Zero-arg builder: returns a fresh module with no LM bound. The worker
    # supplies the LM from its environment at runtime.
    return dspy.ChainOfThought("question -> answer")


qa = dt.program("qa", mode=dt.RunMode.COARSE)
