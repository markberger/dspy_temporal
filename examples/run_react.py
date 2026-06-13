"""Run the fine-mode ReAct agent and print the answer.

Run (with a worker already running):
    uv run python examples/run_react.py "What's the weather in Tokyo?"

In fine mode the worker runs each LM call and each tool call as its own Temporal
activity. Check the Temporal UI (http://localhost:8233) to see the separate
``dspy_lm_call`` / ``dspy_tool_call`` events for a single run; with tracing on,
Phoenix shows the nested per-call spans.

As with the coarse starter, set ``OTEL_EXPORTER_OTLP_ENDPOINT`` to originate the
root span and tie the worker's spans into one trace:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
        uv run --extra tracing python examples/run_react.py "Weather in Tokyo?"
"""

import asyncio
import os
import sys

from react_program import TASK_QUEUE

import dspy_temporal as dt


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Tokyo?"

    interceptors: list = []
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from dspy_temporal.tracing import setup_tracing

        # The starter only originates the root span; it emits no DSPy spans.
        interceptors.append(
            setup_tracing(service_name="dspy-temporal-starter", register_callback=False)
        )

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await dt.connect(address, interceptors=interceptors)
    prediction = await dt.run_program(
        client,
        "weather_agent",
        {"question": question},
        task_queue=TASK_QUEUE,
        mode=dt.RunMode.FINE,
    )
    print("Q:", question)
    print("A:", prediction.answer)


if __name__ == "__main__":
    asyncio.run(main())
