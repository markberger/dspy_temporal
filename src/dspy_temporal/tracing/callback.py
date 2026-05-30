"""A DSPy BaseCallback that emits OpenTelemetry spans (dual-convention).

Registered via ``dspy.settings.callbacks`` (a first-class extension point, no
monkeypatching). Span nesting follows DSPy's ``ACTIVE_CALL_ID`` explicitly (via a
shared call_id -> span map) so it is robust to DSPy's internal worker threads.
The root span parents to whatever OTel context is current -- inside the activity
that is the Temporal activity span (covered by tests/integration/test_tracing_workflow.py).

DSPy swallows callback exceptions (dspy/utils/callback.py), so a bug here can
never break a program run.
"""

from __future__ import annotations

import threading
from typing import Any

from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from . import semconv


class DSPyOTelCallback(BaseCallback):
    def __init__(self, tracer=None, capture_content: bool = False):
        self._tracer = tracer or trace.get_tracer("dspy_temporal.tracing")
        self._capture_content = capture_content
        self._spans: dict[str, Any] = {}
        self._lm_history: dict[str, tuple[Any, int]] = {}
        self._lock = threading.Lock()

    # --- span lifecycle ----------------------------------------------------
    def _start(self, call_id: str, name: str, attributes: dict[str, Any]):
        parent_id = ACTIVE_CALL_ID.get()
        with self._lock:
            parent_span = self._spans.get(parent_id)
        # parent_span is None for the top span -> inherit the current OTel
        # context (the Temporal activity span).
        ctx = set_span_in_context(parent_span) if parent_span is not None else None
        span = self._tracer.start_span(name, context=ctx, attributes=attributes)
        with self._lock:
            self._spans[call_id] = span
        return span

    def _end(self, call_id: str, attributes: dict[str, Any] | None = None, exception=None):
        with self._lock:
            span = self._spans.pop(call_id, None)
        if span is None:
            return
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        if exception is not None:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))
        span.end()

    # --- module (CHAIN) ----------------------------------------------------
    def on_module_start(self, call_id, instance, inputs):
        attrs = semconv.module_attributes(instance)
        if self._capture_content:
            attrs[semconv.INPUT_VALUE] = semconv.safe_json(inputs)
            attrs[semconv.INPUT_MIME_TYPE] = semconv.MIME_JSON
        self._start(call_id, f"dspy.module {type(instance).__name__}", attrs)

    def on_module_end(self, call_id, outputs, exception=None):
        attrs = {}
        if self._capture_content and outputs is not None:
            attrs[semconv.OUTPUT_VALUE] = semconv.safe_json(_to_jsonable(outputs))
            attrs[semconv.OUTPUT_MIME_TYPE] = semconv.MIME_JSON
        self._end(call_id, attrs, exception)

    # --- LM (LLM) ----------------------------------------------------------
    def on_lm_start(self, call_id, instance, inputs):
        attrs = semconv.lm_request_attributes(instance)
        span = self._start(call_id, semconv.lm_span_name(getattr(instance, "model", None)), attrs)
        with self._lock:
            self._lm_history[call_id] = (instance, len(getattr(instance, "history", []) or []))
        if self._capture_content:
            messages = inputs.get("messages") or inputs.get("prompt")
            if messages is not None:
                payload = semconv.safe_json(messages)
                span.add_event("gen_ai.input.messages", {"content": payload})
                span.set_attribute(semconv.INPUT_VALUE, payload)
                span.set_attribute(semconv.INPUT_MIME_TYPE, semconv.MIME_JSON)

    def on_lm_end(self, call_id, outputs, exception=None):
        with self._lock:
            instance, start = self._lm_history.pop(call_id, (None, 0))
        attrs: dict[str, Any] = {}
        if instance is not None:
            new_entries = (getattr(instance, "history", []) or [])[start:]
            if new_entries:
                attrs = semconv.lm_response_attributes(new_entries[-1])
        if self._capture_content and outputs is not None:
            attrs[semconv.OUTPUT_VALUE] = semconv.safe_json(_to_jsonable(outputs))
            attrs[semconv.OUTPUT_MIME_TYPE] = semconv.MIME_JSON
        self._end(call_id, attrs, exception)

    # --- tools (TOOL) ------------------------------------------------------
    def on_tool_start(self, call_id, instance, inputs):
        attrs = semconv.tool_attributes(instance)
        if self._capture_content:
            attrs[semconv.INPUT_VALUE] = semconv.safe_json(inputs)
            attrs[semconv.INPUT_MIME_TYPE] = semconv.MIME_JSON
        name = getattr(instance, "name", None) or type(instance).__name__
        self._start(call_id, f"execute_tool {name}", attrs)

    def on_tool_end(self, call_id, outputs, exception=None):
        attrs = {}
        if self._capture_content and outputs is not None:
            attrs[semconv.OUTPUT_VALUE] = semconv.safe_json(_to_jsonable(outputs))
            attrs[semconv.OUTPUT_MIME_TYPE] = semconv.MIME_JSON
        self._end(call_id, attrs, exception)


def _to_jsonable(obj: Any) -> Any:
    """Best-effort plain-data view of a Prediction/Example for content capture."""
    if hasattr(obj, "toDict"):
        try:
            return obj.toDict()
        except Exception:  # pragma: no cover - defensive
            pass
    return obj
