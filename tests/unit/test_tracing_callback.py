"""Unit tests for DSPyOTelCallback (in-memory exporter, DummyLM, no network)."""

from types import SimpleNamespace

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dspy_temporal.tracing.callback import DSPyOTelCallback


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def _run(tracer, capture_content=False, lm=None):
    cb = DSPyOTelCallback(tracer=tracer, capture_content=capture_content)
    lm = lm or DummyLM([{"reasoning": "r", "answer": "blue"}] * 3)
    with dspy.context(lm=lm, callbacks=[cb], track_usage=True):
        dspy.ChainOfThought("question -> answer")(question="sky?")


def _by_name(spans):
    return {s.name: s for s in spans}


def test_emits_dual_convention_lm_span(tracer, exporter):
    _run(tracer)
    spans = _by_name(exporter.get_finished_spans())
    lm = spans["chat dummy"]
    a = lm.attributes
    # gen_ai.* semconv
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.request.model"] == "dummy"
    assert a["gen_ai.response.model"] == "dummy"
    assert a["gen_ai.usage.input_tokens"] == 0
    assert a["gen_ai.usage.output_tokens"] == 0
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    # OpenInference (Phoenix)
    assert a["openinference.span.kind"] == "LLM"
    assert a["llm.model_name"] == "dummy"
    assert a["llm.token_count.prompt"] == 0
    assert a["llm.token_count.completion"] == 0


def test_module_spans_are_chains_and_nested(tracer, exporter):
    _run(tracer)
    spans = _by_name(exporter.get_finished_spans())
    cot, predict, lm = (
        spans["dspy.module ChainOfThought"],
        spans["dspy.module Predict"],
        spans["chat dummy"],
    )
    assert cot.attributes["openinference.span.kind"] == "CHAIN"
    assert predict.attributes["openinference.span.kind"] == "CHAIN"
    # nesting: chat -> Predict -> ChainOfThought(root)
    assert lm.parent.span_id == predict.context.span_id
    assert predict.parent.span_id == cot.context.span_id
    assert cot.parent is None
    # single trace
    assert lm.context.trace_id == cot.context.trace_id


def test_no_content_by_default(tracer, exporter):
    _run(tracer, capture_content=False)
    lm = _by_name(exporter.get_finished_spans())["chat dummy"]
    assert "input.value" not in lm.attributes
    assert "output.value" not in lm.attributes


def test_content_captured_when_enabled(tracer, exporter):
    _run(tracer, capture_content=True)
    spans = exporter.get_finished_spans()
    lm = _by_name(spans)["chat dummy"]
    assert "input.value" in lm.attributes
    assert "output.value" in lm.attributes
    # content also emitted as a gen_ai span event
    assert any(e.name == "gen_ai.input.messages" for e in lm.events)


def test_tool_span_emitted(tracer, exporter):
    cb = DSPyOTelCallback(tracer=tracer, capture_content=True)
    cb.on_tool_start("c1", SimpleNamespace(name="search"), {"query": "x"})
    cb.on_tool_end("c1", "a result")

    span = next(
        s for s in exporter.get_finished_spans() if s.name.startswith("execute_tool")
    )
    assert span.name == "execute_tool search"
    assert span.attributes["openinference.span.kind"] == "TOOL"
    assert "input.value" in span.attributes
    assert span.attributes["output.value"]


def test_unfinished_call_id_end_is_noop(tracer, exporter):
    # Ending an unknown call_id must not raise (defensive path).
    DSPyOTelCallback(tracer=tracer).on_lm_end("never-started", outputs=None)
    assert exporter.get_finished_spans() == ()


def _history_entry(prompt=7, completion=3):
    return {
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        "response_model": "dummy",
    }


def test_sequential_lm_call_attributes_tokens(tracer, exporter):
    """One entry appended during the call -> usage is attributed to the span."""
    instance = SimpleNamespace(model="dummy", kwargs={}, history=[])
    cb = DSPyOTelCallback(tracer=tracer)
    cb.on_lm_start("c1", instance, {})
    instance.history.append(_history_entry(7, 3))  # this call's entry
    cb.on_lm_end("c1", outputs=["ok"])

    span = next(s for s in exporter.get_finished_spans() if s.name.startswith("chat"))
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 3


def test_concurrent_lm_calls_omit_ambiguous_tokens(tracer, exporter):
    """A shared LM instance interleaves history across threads; when more than one
    entry lands during a call we omit usage rather than report the wrong call's."""
    instance = SimpleNamespace(model="dummy", kwargs={}, history=[])
    cb = DSPyOTelCallback(tracer=tracer)
    cb.on_lm_start("a", instance, {})
    # A concurrent call on the same shared instance appends while "a" is in flight.
    instance.history.append(_history_entry(11, 5))  # not necessarily a's entry
    instance.history.append(_history_entry(99, 42))
    cb.on_lm_end("a", outputs=["ok"])

    span = next(s for s in exporter.get_finished_spans() if s.name.startswith("chat"))
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert "gen_ai.usage.output_tokens" not in span.attributes


def test_exception_records_error_status(tracer, exporter):
    class BoomLM(DummyLM):
        # Raise inside forward() so it happens within the with_callbacks wrapper
        # (overriding __call__ would bypass the LM callbacks entirely).
        def forward(self, *args, **kwargs):
            raise RuntimeError("boom")

    cb = DSPyOTelCallback(tracer=tracer)
    with pytest.raises(RuntimeError):
        with dspy.context(lm=BoomLM([{"answer": "x"}]), callbacks=[cb]):
            dspy.Predict("question -> answer")(question="?")

    lm = next(s for s in exporter.get_finished_spans() if s.name.startswith("chat"))
    assert lm.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in lm.events)
