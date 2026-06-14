"""A DSPy BaseCallback that emits OpenTelemetry spans (dual-convention).

Registered via ``dspy.settings.callbacks`` (a first-class extension point, no
monkeypatching). Span nesting follows DSPy's ``ACTIVE_CALL_ID`` explicitly (via a
shared call_id -> span map). ``ACTIVE_CALL_ID`` is a ContextVar, so nesting holds
on the synchronous path and across ``asyncio.gather`` (asyncio copies the context
into each Task), but NOT across ``dspy.Parallel``'s ``ThreadPoolExecutor`` -- which
does not copy contextvars, so parallel sub-calls orphan into new trace roots. For a
correct trace tree under concurrency, drive the program via the async interface
(``acall``/``aforward`` + ``asyncio.gather``) rather than ``dspy.Parallel``; the
coarse activity does this by default. See docs/tracing-design.md ("Concurrency &
nesting").

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
            self._spans[call_id] = span
        return span

    def _end(
        self, call_id: str, attributes: dict[str, Any] | None = None, exception=None
    ):
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

    # --- content capture (opt-in) ------------------------------------------
    def _input_content_attrs(self, value: Any) -> dict[str, Any]:
        if not self._capture_content:
            return {}
        return {
            semconv.INPUT_VALUE: semconv.safe_json(value),
            semconv.INPUT_MIME_TYPE: semconv.MIME_JSON,
        }

    def _output_content_attrs(self, outputs: Any) -> dict[str, Any]:
        if not self._capture_content or outputs is None:
            return {}
        return {
            semconv.OUTPUT_VALUE: semconv.safe_json(_to_jsonable(outputs)),
            semconv.OUTPUT_MIME_TYPE: semconv.MIME_JSON,
        }

    # --- module (CHAIN) ----------------------------------------------------
    def on_module_start(self, call_id, instance, inputs):
        attrs = semconv.module_attributes()
        attrs.update(self._input_content_attrs(inputs))
        self._start(call_id, f"dspy.module {type(instance).__name__}", attrs)

    def on_module_end(self, call_id, outputs, exception=None):
        self._end(call_id, self._output_content_attrs(outputs), exception)

    # --- LM (LLM) ----------------------------------------------------------
    def on_lm_start(self, call_id, instance, inputs):
        attrs = semconv.lm_request_attributes(instance)
        span = self._start(
            call_id, semconv.lm_span_name(getattr(instance, "model", None)), attrs
        )
        with self._lock:
            self._lm_history[call_id] = (
                instance,
                len(getattr(instance, "history", []) or []),
            )
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
            # The coarse worker shares one LM instance across concurrent
            # activities, so its `history` list interleaves entries from calls
            # racing in other threads. Only attribute usage/cost/model when
            # exactly one entry was appended during this call -- otherwise we
            # cannot tell which entry is ours, so we omit the response attributes
            # rather than risk reporting another call's tokens/cost.
            if len(new_entries) == 1:
                attrs = semconv.lm_response_attributes(new_entries[0])
        attrs.update(self._output_content_attrs(outputs))
        self._end(call_id, attrs, exception)

    # --- tools (TOOL) ------------------------------------------------------
    def on_tool_start(self, call_id, instance, inputs):
        attrs = semconv.tool_attributes()
        attrs.update(self._input_content_attrs(inputs))
        name = getattr(instance, "name", None) or type(instance).__name__
        self._start(call_id, f"execute_tool {name}", attrs)

    def on_tool_end(self, call_id, outputs, exception=None):
        self._end(call_id, self._output_content_attrs(outputs), exception)


def _to_jsonable(obj: Any) -> Any:
    """Best-effort plain-data view of a Prediction/Example for content capture."""
    if hasattr(obj, "toDict"):
        try:
            return obj.toDict()
        except Exception:  # noqa: S110  # pragma: no cover - intentional fallback below
            pass
    return obj
