"""End-to-end tracing: the activity span parents the DSPy spans (one trace).

Validates the client-only interceptor registration (no duplicate spans) and the
ThreadPoolExecutor context bridge proven by scripts/tracing_spike.py.
"""

import uuid

import dspy
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from temporalio.testing import WorkflowEnvironment

import dspy_temporal as dt
from dspy_temporal.config import RunConfig
from dspy_temporal.converter import data_converter
from dspy_temporal.tracing import setup_tracing


@pytest.fixture(scope="module")
def otel():
    """One in-memory provider for the module (global provider can only be set once)."""
    import opentelemetry.trace as trace_api

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    return provider, exporter


@pytest.mark.asyncio
async def test_activity_span_parents_dspy_spans(otel, dummy_lm):
    provider, exporter = otel
    exporter.clear()

    interceptor = setup_tracing(tracer_provider=provider, set_global=False)
    dt.register_program("qa", lambda: dspy.ChainOfThought("question -> answer"))
    dt.set_worker_lm(dummy_lm)

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter,
        interceptors=[interceptor],  # client only; worker inherits it
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client, "qa", {"question": "sky?"}, task_queue=task_queue
            )

    assert pred.answer == "blue"
    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}

    activity_ids = {s.context.span_id for s in spans if "RunActivity" in s.name}
    assert activity_ids, "no Temporal activity span"

    cot = by_name["dspy.module ChainOfThought"]
    lm = by_name["chat dummy"]

    # Root dspy span parents to the activity span -> one unified trace.
    assert cot.parent is not None and cot.parent.span_id in activity_ids
    assert lm.context.trace_id == cot.context.trace_id
    # gen_ai + OpenInference attributes survived the Temporal boundary.
    assert lm.attributes["gen_ai.request.model"] == "dummy"
    assert lm.attributes["openinference.span.kind"] == "LLM"

    # Client-only registration -> DSPy spans emitted exactly once (no duplicates).
    assert sum(1 for s in spans if s.name == "chat dummy") == 1
    assert sum(1 for s in spans if s.name == "dspy.module ChainOfThought") == 1
