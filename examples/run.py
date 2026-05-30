"""Start the example QA program as a durable workflow and print the result.

Run (with a worker already running):
    uv run python examples/run.py "Why is the sky blue?"
"""

import asyncio
import sys

import dspy_temporal as dt

from qa_program import TASK_QUEUE  # noqa: E402


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"

    client = await dt.connect("localhost:7233")
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
