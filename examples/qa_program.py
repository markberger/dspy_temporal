"""Shared program definition imported by both the worker and the starter.

Defining the program here (and importing this module from the worker) is how the
worker's registry learns about the program builder.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"


def build_qa() -> dspy.Module:
    # Zero-arg builder: returns a fresh module with no LM bound. The worker
    # supplies the LM from its environment at runtime.
    return dspy.ChainOfThought("question -> answer")


qa = dt.deploy(
    build_qa,
    name="qa",
    task_queue=TASK_QUEUE,
)
