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

import dspy
from compose_program import ResearchWorkflow
from compose_refs import triage_agent
from instance_program import prototype, qa_instance
from qa_program import TASK_QUEUE, build_qa, qa
from react_program import build_weather_agent, weather_agent
from temporalio.worker import Worker
from two_lm_program import TwoLMQA, two_lm_qa

import dspy_temporal as dt


async def main() -> None:
    dt.configure_lm_from_env()
    # Bind each declared program to its implementation (worker-side only).
    qa.bind(build_qa)
    weather_agent.bind(build_weather_agent)
    two_lm_qa.bind(TwoLMQA)
    qa_instance.bind(prototype)
    triage_agent.bind(lambda: dspy.ChainOfThought("question -> answer"))

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
