"""Tests for setup_tracing wiring and content-capture resolution."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from temporalio.contrib.opentelemetry import TracingInterceptor

from dspy_temporal import config as core_config
from dspy_temporal.tracing import setup_tracing
from dspy_temporal.tracing.callback import DSPyOTelCallback
from dspy_temporal.tracing.config import build_tracer_provider, resolve_capture_content


def test_setup_tracing_builds_provider_from_exporter():
    # tracer_provider=None + an explicit exporter -> we build a provider (no OTLP).
    exporter = InMemorySpanExporter()
    interceptor = setup_tracing(exporter=exporter, set_global=False)
    assert interceptor is not None
    assert isinstance(core_config.get_tracing_callback(), DSPyOTelCallback)


def test_build_tracer_provider_with_exporter():
    provider = build_tracer_provider("svc", InMemorySpanExporter())
    assert provider.get_tracer("x") is not None


def test_setup_tracing_does_not_override_installed_global(monkeypatch):
    """With a global SDK provider already installed and an explicit exporter, we
    build our own provider but must NOT try to promote it to global -- OTel
    forbids overriding, so that call would be a confusing no-op."""
    from opentelemetry import trace as trace_api

    installed = TracerProvider()  # a real SDK provider already in place
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: installed)
    set_calls = []
    monkeypatch.setattr(trace_api, "set_tracer_provider", set_calls.append)

    interceptor = setup_tracing(exporter=InMemorySpanExporter(), set_global=True)

    assert isinstance(interceptor, TracingInterceptor)
    assert set_calls == []  # never attempted to override the installed global


def test_clear_tracing_callback_roundtrip():
    setup_tracing(
        tracer_provider=TracerProvider(), set_global=False, register_callback=True
    )
    assert core_config.get_tracing_callback() is not None
    core_config.clear_tracing_callback()
    assert core_config.get_tracing_callback() is None


def test_setup_tracing_registers_callback_and_returns_interceptor():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))

    interceptor = setup_tracing(tracer_provider=provider, set_global=False)

    assert isinstance(interceptor, TracingInterceptor)
    assert isinstance(core_config.get_tracing_callback(), DSPyOTelCallback)


def test_setup_tracing_can_skip_callback_registration():
    provider = TracerProvider()
    interceptor = setup_tracing(
        tracer_provider=provider, set_global=False, register_callback=False
    )
    assert isinstance(interceptor, TracingInterceptor)
    assert core_config.get_tracing_callback() is None


def test_capture_content_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    assert resolve_capture_content(False) is False
    assert resolve_capture_content(True) is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("span_and_event", True),
        ("false", False),
        ("no", False),
    ],
)
def test_capture_content_env_values(monkeypatch, value, expected):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", value)
    assert resolve_capture_content(None) is expected


def test_capture_content_default_off(monkeypatch):
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    assert resolve_capture_content(None) is False
