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


def test_setup_tracing_reuses_installed_global_provider(monkeypatch):
    """tracer_provider=None + no exporter + an SDK provider already installed:
    reuse that provider as-is (don't build or override)."""
    from opentelemetry import trace as trace_api

    installed = TracerProvider()
    installed.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: installed)
    set_calls = []
    monkeypatch.setattr(trace_api, "set_tracer_provider", set_calls.append)

    interceptor = setup_tracing(set_global=True)

    assert isinstance(interceptor, TracingInterceptor)
    # The reused provider became our flush handle; we never set a new global.
    assert core_config.get_tracing_shutdown() == installed.force_flush
    assert set_calls == []


def test_setup_tracing_builds_and_promotes_global(monkeypatch):
    """No SDK provider installed yet -> build one and promote it to the global."""
    from opentelemetry import trace as trace_api

    # A non-SDK default (e.g. the proxy provider) is what's installed pre-setup.
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: object())
    set_calls = []
    monkeypatch.setattr(trace_api, "set_tracer_provider", set_calls.append)

    interceptor = setup_tracing(exporter=InMemorySpanExporter(), set_global=True)

    assert isinstance(interceptor, TracingInterceptor)
    assert len(set_calls) == 1  # promoted the freshly built provider to global


def test_setup_tracing_sets_global_for_passed_provider(monkeypatch):
    """An explicit tracer_provider + set_global=True promotes it to the global."""
    from opentelemetry import trace as trace_api

    set_calls = []
    monkeypatch.setattr(trace_api, "set_tracer_provider", set_calls.append)
    provider = TracerProvider()

    interceptor = setup_tracing(tracer_provider=provider, set_global=True)

    assert isinstance(interceptor, TracingInterceptor)
    assert set_calls == [provider]


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


# --- #17 item 4: span flush on worker stop ----------------------------------


def test_setup_tracing_registers_flush_handle():
    """By default setup_tracing registers the provider's force_flush as the
    worker-stop shutdown hook, and calling it flushes buffered spans."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))

    setup_tracing(tracer_provider=provider, set_global=False)

    flush = core_config.get_tracing_shutdown()
    assert flush == provider.force_flush
    assert flush() is True  # force_flush succeeds (nothing buffered to drain)


def test_setup_tracing_opt_out_leaves_shutdown_unset():
    """flush_on_worker_stop=False leaves no shutdown hook registered (for a
    provider the caller manages/reuses elsewhere)."""
    provider = TracerProvider()
    setup_tracing(
        tracer_provider=provider, set_global=False, flush_on_worker_stop=False
    )
    assert core_config.get_tracing_shutdown() is None


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
