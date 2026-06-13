"""Worker wiring via ``DSPyPlugin`` -- the declarative alternative to build_worker.

``dt.build_worker`` is the one-call path. If you already construct your own
``temporalio.worker.Worker`` (custom interceptors, tuning, extra workflows/
activities of your own), add DSPy support with the plugin instead::

    Worker(client, task_queue=..., plugins=[dt.DSPyPlugin()])

The plugin contributes the same four activities, both generic workflows, and the
DSPy sandbox runner -- extending (never overwriting) anything you pass explicitly.
Pass your own composed workflows via ``DSPyPlugin(extra_workflows=[...])``.

Run:
    uv run python examples/worker_plugin.py
"""

import asyncio
import os

# Importing these registers the program builders + the composed user workflow.
import compose_program  # noqa: F401  (registers "compose_qa" + ResearchWorkflow)
import deploy_instance  # noqa: F401  (registers "qa_instance" from a live instance)
import react_program  # noqa: F401  (registers "weather_agent")
import two_lm_program  # noqa: F401  (registers "two_lm_qa")
from compose_program import ResearchWorkflow
from qa_program import TASK_QUEUE
from temporalio.worker import Worker

import dspy_temporal as dt


async def main() -> None:
    dt.configure_lm_from_env()
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await dt.connect(address)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        plugins=[dt.DSPyPlugin(extra_workflows=[ResearchWorkflow])],
    )
    print(
        f"Plugin worker running on task queue {TASK_QUEUE!r} (Temporal at {address})."
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
