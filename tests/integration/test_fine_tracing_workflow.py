"""Fine-mode tracing: one isolated LM span per call, parented to its activity.

The fine workflow runs the program with ``callbacks=[]`` (no span emission in
workflow code), so *every* DSPy span originates inside an activity. This proves:

- each ``dspy_lm_call`` activity emits exactly one ``chat <model>`` LLM span,
  parented to that activity's ``RunActivity`` span;
- each LM span carries its own ``gen_ai.usage.*`` tokens -- the per-call
  attribution that coarse mode must omit under concurrency (the documented
  shared-history caveat), because each fine LM call runs on an isolated
  ``worker_lm.copy()`` with a one-entry history;
- a ReAct tool call emits an ``execute_tool <name>`` span from the
  ``dspy_tool_call`` activity;
- no ``dspy.module`` span is emitted at all (the module orchestrates in the
  workflow, where spans are deliberately suppressed).

A local (non-global) ``TracerProvider`` is used so this coexists with the coarse
tracing test, which installs the global provider (OTel forbids overriding it).
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


@pytest.fixture
def tracing():
    """A local in-memory provider + the Temporal interceptor wired to it.

    ``set_global=False`` keeps OTel's single global provider free for the coarse
    tracing test; both the interceptor and the DSPy callback use this provider,
    so all spans (Temporal's ``RunActivity`` + DSPy's) land in one exporter.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    interceptor = setup_tracing(tracer_provider=provider, set_global=False)
    return interceptor, exporter


def _activity_span_ids(spans, activity_type=None):
    """Span ids of the worker-side ``RunActivity`` spans (optionally one type)."""
    ids = set()
    for s in spans:
        if not s.name.startswith("RunActivity"):
            continue
        if activity_type is not None and activity_type not in s.name:
            continue
        ids.add(s.context.span_id)
    return ids


@pytest.mark.asyncio
async def test_fine_lm_span_parents_to_its_activity_with_usage(tracing, dummy_lm):
    """A fine ChainOfThought: one LM call -> one activity -> one LM span that
    carries its own usage and parents to the ``dspy_lm_call`` activity span."""
    interceptor, exporter = tracing
    dt.register_program("qa", lambda: dspy.ChainOfThought("question -> answer"))
    dt.set_worker_lm(dummy_lm)

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter,
        interceptors=[interceptor],  # client only; the worker inherits it
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                "qa",
                {"question": "sky?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert pred.answer == "blue"
    spans = exporter.get_finished_spans()

    # Exactly one LM span, emitted inside the dspy_lm_call activity.
    lm_spans = [s for s in spans if s.name == "chat dummy"]
    assert len(lm_spans) == 1
    lm = lm_spans[0]

    lm_activity_ids = _activity_span_ids(spans, "dspy_lm_call")
    assert lm_activity_ids, "no dspy_lm_call activity span"
    assert lm.parent is not None
    assert lm.parent.span_id in lm_activity_ids

    # Per-call usage attribution -- present on the span (the coarse concurrency
    # caveat is gone: the isolated lm.copy() has a single, unambiguous entry).
    assert "gen_ai.usage.total_tokens" in lm.attributes
    assert lm.attributes["gen_ai.request.model"] == "dummy"
    assert lm.attributes["openinference.span.kind"] == "LLM"

    # The module orchestrates in the workflow (callbacks=[]) -> no module span.
    assert not any(s.name.startswith("dspy.module") for s in spans)


@pytest.mark.asyncio
async def test_fine_react_emits_lm_and_tool_spans_per_activity(tracing, fine_react):
    """A fine ReAct run: every LM call and the tool call each emit their own span
    from their own activity; no orchestration span leaks from the workflow."""
    interceptor, exporter = tracing

    task_queue = f"tq-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=data_converter,
        interceptors=[interceptor],
    ) as env:
        worker = dt.build_worker(env.client, config=RunConfig(task_queue=task_queue))
        async with worker:
            pred = await dt.run_program(
                env.client,
                fine_react.name,
                {"question": "What's the weather in Tokyo?"},
                task_queue=task_queue,
                mode=dt.RunMode.FINE,
            )

    assert "sunny" in pred.answer.lower()
    spans = exporter.get_finished_spans()

    # One LM span per LM call (2 react steps + 1 extract), each its own activity.
    lm_spans = [s for s in spans if s.name == "chat dummy"]
    assert len(lm_spans) == 3
    lm_activity_ids = _activity_span_ids(spans, "dspy_lm_call")
    # Each LM span parents to a dspy_lm_call activity, and each carries its own
    # usage (isolated copies -> unambiguous per-call attribution).
    for lm in lm_spans:
        assert lm.parent is not None
        assert lm.parent.span_id in lm_activity_ids
        assert lm.attributes["gen_ai.usage.total_tokens"] == 3

    # The single tool call emits an execute_tool span from the tool activity.
    tool_spans = [s for s in spans if s.name == "execute_tool _weather_tool"]
    assert len(tool_spans) == 1
    tool = tool_spans[0]
    tool_activity_ids = _activity_span_ids(spans, "dspy_tool_call")
    assert tool_activity_ids, "no dspy_tool_call activity span"
    assert tool.parent is not None
    assert tool.parent.span_id in tool_activity_ids
    assert tool.attributes["openinference.span.kind"] == "TOOL"

    # No module/orchestration span -- all spans came from activities.
    assert not any(s.name.startswith("dspy.module") for s in spans)
