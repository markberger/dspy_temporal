"""Run a worker that serves the example QA program.

Works both on the host and inside the Docker Compose stack:
    - Connects to ``TEMPORAL_ADDRESS`` (default ``localhost:7233``).
    - If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, enables tracing and registers the
      interceptor on the client (the worker inherits it) so LLM spans are exported
      (e.g. to Phoenix over OTLP gRPC).

Prereqs:
    - A Temporal server:  temporal server start-dev   (or the compose ``temporal`` service)
    - An LM configured via env, e.g.:
        export DSPY_LM_MODEL=openrouter/openai/gpt-5-nano
        export OPENROUTER_API_KEY=sk-or-...

Run:
    uv run python examples/worker.py
"""

import asyncio
import os

import dspy

# Each module declares a side-effect-free program reference (dt.program(...)); the
# worker attaches the implementation below with ref.bind(impl). The refs:
#   - qa             -> "qa" (coarse mode)
#   - weather_agent  -> "weather_agent" (fine mode; per-LM/per-tool activities)
#   - two_lm_qa      -> "two_lm_qa" (fine mode; per-predictor multi-LM)
#   - qa_instance    -> "qa_instance" (a live dspy.Module instance)
#   - triage_agent   -> "compose_qa", composed inside ResearchWorkflow
from compose_program import ResearchWorkflow
from compose_refs import triage_agent
from instance_program import prototype, qa_instance
from qa_program import TASK_QUEUE, build_qa, qa
from react_program import build_weather_agent, weather_agent
from two_lm_program import TwoLMQA, two_lm_qa

import dspy_temporal as dt


async def _connect_with_retry(address: str, *, interceptors, attempts: int = 30):
    """Connect to Temporal, retrying while the dev server is still booting.

    Compose ``depends_on`` only waits for container start, not frontend readiness,
    so the first few connects can fail with a transient error.
    """
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return await dt.connect(address, interceptors=interceptors)
        except Exception as exc:  # retry any connect failure
            last_exc = exc
            await asyncio.sleep(1)
    raise RuntimeError(f"Could not connect to Temporal at {address}") from last_exc


async def main() -> None:
    # Configure the LM from the environment (worker-side only; never serialized).
    dt.configure_lm_from_env()

    # Bind each declared program to its implementation. This is the heavy,
    # side-effecting step (it populates the process registry) and belongs on the
    # worker -- never in a workflow file.
    qa.bind(build_qa)
    weather_agent.bind(build_weather_agent)
    two_lm_qa.bind(TwoLMQA)
    qa_instance.bind(prototype)
    triage_agent.bind(lambda: dspy.ChainOfThought("question -> answer"))

    interceptors: list = []
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from dspy_temporal.tracing import setup_tracing

        # Registers the DSPy span-emitting callback (worker side) and returns the
        # Temporal interceptor to attach to the client.
        interceptors.append(setup_tracing(service_name="dspy-temporal-worker"))
        print(
            "Tracing enabled -> exporting to", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        )

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await _connect_with_retry(address, interceptors=interceptors)
    # extra_workflows serves the user-authored ResearchWorkflow alongside the
    # two generic DSPy workflows.
    worker = dt.build_worker(
        client,
        task_queue=TASK_QUEUE,
        extra_workflows=[ResearchWorkflow],
    )
    print(
        f"Worker running on task queue {TASK_QUEUE!r} (Temporal at {address}). Ctrl-C to exit."
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
