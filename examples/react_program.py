"""A fine-mode ReAct agent: each LM call and tool call is its own activity.

Deploying with ``mode="fine"`` makes the worker orchestrate the ReAct loop in
the workflow and run every model call (and every ``get_weather`` call) as a
separate, independently-retried Temporal activity. In the Temporal UI you'll see
distinct ``dspy_lm_call`` / ``dspy_tool_call`` events; on a crash the run resumes
from the last completed one.

The tool function body runs in an activity, so it may do real I/O (here it just
returns a string). The builder runs in the workflow, so it only *constructs*
dspy objects -- no network/file/DB at build time.

Imported by ``examples/worker.py`` so the builder registers at worker startup.
"""

import dspy

import dspy_temporal as dt

TASK_QUEUE = "dspy-temporal-example"


def get_weather(city: str) -> str:
    """Return a short weather report for a city."""
    # Tool bodies run inside the dspy_tool_call activity, so real I/O (HTTP, DB)
    # is allowed here. Kept trivial so the example needs no network.
    return f"The weather in {city} is sunny and 22°C."


def build_weather_agent() -> dspy.Module:
    # Zero-arg builder, run in the workflow: only construct dspy objects, no I/O.
    # The worker supplies the LM from its environment at runtime.
    return dspy.ReAct("question -> answer", tools=[get_weather])


weather_agent = dt.deploy_module(
    "weather_agent",
    build_weather_agent,
    config=dt.RunConfig(task_queue=TASK_QUEUE, mode="fine"),
)
