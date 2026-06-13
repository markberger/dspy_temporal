"""Dual-convention attribute mapping: OTel GenAI semconv + OpenInference.

This is the single place that knows attribute names, so the (still-evolving)
gen_ai spec churn is contained here. It builds plain dicts of non-None
attributes from DSPy data; it imports no OpenTelemetry (the callback applies
these to spans).
"""

from __future__ import annotations

import json
from typing import Any

# --- OTel GenAI semantic conventions ---------------------------------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_COST = "gen_ai.usage.cost"

# --- OpenInference conventions (for Arize Phoenix) -------------------------
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
LLM_MODEL_NAME = "llm.model_name"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
INPUT_MIME_TYPE = "input.mime_type"
OUTPUT_MIME_TYPE = "output.mime_type"
MIME_JSON = "application/json"

# OpenInference span kinds
KIND_LLM = "LLM"
KIND_CHAIN = "CHAIN"
KIND_TOOL = "TOOL"

_PARAM_KEYS = ("temperature", "max_tokens", "top_p")


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(obj)


def provider_from_model(model: str | None) -> str | None:
    """`openai/gpt-4o-mini` -> `openai`; returns None when no provider prefix."""
    if not model or "/" not in model:
        return None
    return model.split("/", 1)[0]


def lm_span_name(model: str | None) -> str:
    return f"chat {model}" if model else "chat"


def module_attributes() -> dict[str, Any]:
    return {OPENINFERENCE_SPAN_KIND: KIND_CHAIN}


def tool_attributes() -> dict[str, Any]:
    return {OPENINFERENCE_SPAN_KIND: KIND_TOOL}


def lm_request_attributes(instance: Any) -> dict[str, Any]:
    """Request-side attributes from the LM instance (model + parameters)."""
    model = getattr(instance, "model", None)
    attrs: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: "chat",
        OPENINFERENCE_SPAN_KIND: KIND_LLM,
    }
    if model:
        attrs[GEN_AI_REQUEST_MODEL] = model
        attrs[LLM_MODEL_NAME] = model
        provider = provider_from_model(model)
        if provider:
            attrs[GEN_AI_SYSTEM] = provider
            attrs[GEN_AI_PROVIDER_NAME] = provider

    kwargs = dict(getattr(instance, "kwargs", {}) or {})
    if kwargs.get("temperature") is not None:
        attrs[GEN_AI_REQUEST_TEMPERATURE] = kwargs["temperature"]
    if kwargs.get("max_tokens") is not None:
        attrs[GEN_AI_REQUEST_MAX_TOKENS] = kwargs["max_tokens"]
    if kwargs.get("top_p") is not None:
        attrs[GEN_AI_REQUEST_TOP_P] = kwargs["top_p"]

    params = {
        k: v for k, v in kwargs.items() if not k.startswith("api") and v is not None
    }
    if params:
        attrs[LLM_INVOCATION_PARAMETERS] = safe_json(params)
    return attrs


def _finish_reasons(response: Any) -> list[str] | None:
    try:
        reasons = [
            c.finish_reason
            for c in response.choices
            if getattr(c, "finish_reason", None)
        ]
    except Exception:
        return None
    return reasons or None


def lm_response_attributes(history_entry: dict[str, Any]) -> dict[str, Any]:
    """Response-side attributes from the LM history entry added during the call."""
    attrs: dict[str, Any] = {}
    response_model = history_entry.get("response_model")
    if response_model:
        attrs[GEN_AI_RESPONSE_MODEL] = response_model

    usage = history_entry.get("usage") or {}
    inp = usage.get("prompt_tokens", usage.get("input_tokens"))
    out = usage.get("completion_tokens", usage.get("output_tokens"))
    tot = usage.get("total_tokens")
    if inp is not None:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = inp
        attrs[LLM_TOKEN_COUNT_PROMPT] = inp
    if out is not None:
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = out
        attrs[LLM_TOKEN_COUNT_COMPLETION] = out
    if tot is not None:
        attrs[GEN_AI_USAGE_TOTAL_TOKENS] = tot
        attrs[LLM_TOKEN_COUNT_TOTAL] = tot

    reasons = _finish_reasons(history_entry.get("response"))
    if reasons:
        attrs[GEN_AI_RESPONSE_FINISH_REASONS] = reasons

    cost = history_entry.get("cost")
    if cost is not None:
        attrs[GEN_AI_USAGE_COST] = cost
    return attrs
