"""Run the two-LM fine-mode program and print the answer + per-LM usage.

Run (with a worker already running and OPENROUTER_API_KEY set on the worker):
    uv run python examples/run_two_lm.py "Why is the sky blue?"

In fine mode each predictor's LM call is its own ``dspy_lm_call`` activity, so a
single run uses *both* bound models -- visible here as two distinct keys in the
printed ``lm_usage`` (and as two LM spans in the Temporal UI / Phoenix).

As with the other starters, set ``OTEL_EXPORTER_OTLP_ENDPOINT`` to originate the
root span and tie the worker's spans into one trace:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
        uv run --extra tracing python examples/run_two_lm.py "Why is the sky blue?"
"""

import asyncio
import json
import os
import sys

import dspy_temporal as dt

from two_lm_program import TASK_QUEUE  # noqa: E402


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"

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
        "two_lm_qa",
        {"question": question},
        task_queue=TASK_QUEUE,
        mode=dt.RunMode.FINE,
    )

    print("Q:", question)
    print("A:", prediction.answer)
    # The proof of the two-LM split: usage is keyed by each predictor's model.
    print("lm_usage:", json.dumps(prediction.get_lm_usage(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
