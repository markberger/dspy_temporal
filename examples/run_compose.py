"""Start the user-authored ResearchWorkflow (which composes a deployed program).

Run (with a worker already running that serves ResearchWorkflow):
    uv run python examples/run_compose.py "Why is the sky blue?"

Unlike ``run.py`` (which calls ``dt.run_program`` to start one of our generic
program workflows), this starts the *user's* workflow directly -- ``agent.run()``
is dispatched inside it. The worker must register ResearchWorkflow, e.g. via
``build_worker(..., extra_workflows=[ResearchWorkflow])`` (see examples/worker.py).
"""

import asyncio
import os
import sys
import uuid

from compose_program import TASK_QUEUE, ResearchWorkflow

import dspy_temporal as dt


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await dt.connect(address)
    answer = await client.execute_workflow(
        ResearchWorkflow.run,
        question,
        id=f"research-{uuid.uuid4().hex[:12]}",
        task_queue=TASK_QUEUE,
    )
    print("Q:", question)
    print("A:", answer)


if __name__ == "__main__":
    asyncio.run(main())
