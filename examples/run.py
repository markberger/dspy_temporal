"""Start the example QA program as a durable workflow and print the result.

Run (with a worker already running):
    uv run python examples/run.py "Why is the sky blue?"

To get a single end-to-end trace, the *initiator* must start the root span and
propagate context across the workflow boundary (standard distributed-tracing). So
when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, this enables tracing and registers the
interceptor on the client; the ``StartWorkflow`` span it emits seeds the trace that
``RunWorkflow``, ``StartActivity``, ``RunActivity`` (and the DSPy spans) all share:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
        uv run --extra tracing python examples/run.py "Why is the sky blue?"
"""

import asyncio
import os
import sys

from qa_program import TASK_QUEUE  # noqa: E402

import dspy_temporal as dt


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"

    interceptors: list = []
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from dspy_temporal.tracing import setup_tracing

        # Starter emits no DSPy spans -- it only needs to originate the root span
        # and propagate context, so skip the DSPy callback.
        interceptors.append(
            setup_tracing(service_name="dspy-temporal-starter", register_callback=False)
        )

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await dt.connect(address, interceptors=interceptors)
    prediction = await dt.run_program(
        client,
        "qa",
        {"question": question},
        task_queue=TASK_QUEUE,
    )
    print("Q:", question)
    print("A:", prediction.answer)
    if getattr(prediction, "reasoning", None):
        print("reasoning:", prediction.reasoning)


if __name__ == "__main__":
    asyncio.run(main())
