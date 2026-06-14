"""OpenTelemetry tracing for DSPy-on-Temporal (dual-emit: gen_ai + OpenInference).

Opt-in: import this subpackage only when you want tracing (it pulls in
OpenTelemetry). The core ``dspy_temporal`` package never imports it.

Usage (per Temporal docs, register the interceptor on the CLIENT only):

    from dspy_temporal.tracing import setup_tracing
    interceptor = setup_tracing(service_name="my-worker")
    client = await dt.connect("localhost:7233", interceptors=[interceptor])
    worker = dt.build_worker(client, task_queue=...)   # inherits the interceptor
"""

from __future__ import annotations

from .callback import DSPyOTelCallback
from .config import build_tracer_provider, resolve_capture_content

__all__ = ["DSPyOTelCallback", "setup_tracing"]


def setup_tracing(
    service_name: str = "dspy-temporal",
    *,
    exporter=None,
    tracer_provider=None,
    capture_content: bool | None = None,
    register_callback: bool = True,
    set_global: bool = True,
    always_create_workflow_spans: bool = True,
):
    """Configure tracing and return the Temporal ``TracingInterceptor``.

    Pass the returned interceptor to ``dt.connect(..., interceptors=[it])`` (the
    worker inherits it from the client). Registers a ``DSPyOTelCallback`` so the
    program activity emits LLM spans.

    Args:
        service_name: ``service.name`` resource attribute. Only applied when we
            build a provider; ignored if an existing provider is reused.
        exporter: a span exporter; defaults to OTLP gRPC when we build the provider.
            Pass an ``InMemorySpanExporter`` in tests. Ignored when an existing SDK
            provider is reused (i.e. ``tracer_provider`` is None and a global SDK
            provider is already installed and no ``exporter`` is given).
        tracer_provider: use this provider instead of building one.
        capture_content: capture prompt/completion text. ``None`` -> read the
            ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` env var (default OFF).
        register_callback: register the DSPy span-emitting callback (worker side).
        set_global: promote our provider to the global OTel tracer provider. A
            no-op if a global provider is already installed (OTel forbids
            overriding it).
        always_create_workflow_spans: create workflow/activity spans even when the
            client call has no parent span, so each execution is a full trace.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from temporalio.contrib.opentelemetry import TracingInterceptor

    from .. import config as core_config

    capture = resolve_capture_content(capture_content)

    if tracer_provider is None:
        existing = trace.get_tracer_provider()
        if isinstance(existing, SDKTracerProvider) and exporter is None:
            # Reuse the user's already-configured provider as-is (their
            # service_name stands; we don't override it).
            tracer_provider = existing
        else:
            tracer_provider = build_tracer_provider(service_name, exporter)
            # Promote to global only if nothing real is installed yet. OTel
            # refuses to override an existing provider (and logs a warning), so
            # calling set_tracer_provider over an SDK provider would be a
            # confusing no-op -- the interceptor/callback still use the provider
            # we just built and return, but the global stays whatever was there.
            if set_global and not isinstance(existing, SDKTracerProvider):
                trace.set_tracer_provider(tracer_provider)
    elif set_global:
        trace.set_tracer_provider(tracer_provider)

    tracer = tracer_provider.get_tracer("dspy_temporal.tracing")

    if register_callback:
        core_config.set_tracing_callback(
            DSPyOTelCallback(tracer=tracer, capture_content=capture)
        )

    return TracingInterceptor(
        tracer=tracer, always_create_workflow_spans=always_create_workflow_spans
    )
