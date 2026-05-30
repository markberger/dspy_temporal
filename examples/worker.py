"""Run a worker that serves the example QA program.

Prereqs:
    - A Temporal dev server:  temporal server start-dev
    - An LM configured via env, e.g.:
        export DSPY_LM_MODEL=openai/gpt-4o-mini
        export OPENAI_API_KEY=sk-...

Run:
    uv run python examples/worker.py
"""

import asyncio

import dspy_temporal as dt

# Importing this registers the "qa" program builder in the process registry.
from qa_program import TASK_QUEUE  # noqa: E402


async def main() -> None:
    # Configure the LM from the environment (worker-side only; never serialized).
    dt.configure_lm_from_env()

    client = await dt.connect("localhost:7233")
    worker = dt.build_worker(client, config=dt.RunConfig(task_queue=TASK_QUEUE))
    print(f"Worker running on task queue {TASK_QUEUE!r}. Ctrl-C to exit.")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
