"""Tracing config: content-capture resolution and tracer-provider construction."""

from __future__ import annotations

import os

CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_TRUTHY = {"true", "1", "span_only", "event_only", "span_and_event"}


def resolve_capture_content(explicit: bool | None) -> bool:
    """Resolve whether to capture message content.

    Explicit argument wins; otherwise read the OTel-standard env var. Default is
    OFF (metadata only) for privacy.
    """
    if explicit is not None:
        return explicit
    val = os.environ.get(CAPTURE_ENV)
    if val is None:
        return False
    return val.strip().lower() in _TRUTHY


def build_tracer_provider(service_name: str, exporter=None):
    """Build an SDK TracerProvider exporting to ``exporter`` (OTLP by default)."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is None:  # pragma: no cover - real OTLP exporter needs a backend
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider
