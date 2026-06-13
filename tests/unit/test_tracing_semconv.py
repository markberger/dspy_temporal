"""Unit tests for the dual-emit semconv mapping (pure functions, no OTel)."""

from types import SimpleNamespace

from dspy_temporal.tracing import semconv


def test_provider_from_model():
    assert semconv.provider_from_model("openai/gpt-4o-mini") == "openai"
    assert semconv.provider_from_model("gpt-4o-mini") is None
    assert semconv.provider_from_model(None) is None


def test_lm_span_name():
    assert semconv.lm_span_name("openai/gpt-4o") == "chat openai/gpt-4o"
    assert semconv.lm_span_name(None) == "chat"


def test_lm_request_attributes_full():
    instance = SimpleNamespace(
        model="openai/gpt-4o-mini",
        kwargs={
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 0.9,
            "api_key": "secret",
        },
    )
    a = semconv.lm_request_attributes(instance)
    assert a["gen_ai.request.model"] == "openai/gpt-4o-mini"
    assert a["gen_ai.system"] == "openai"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.request.temperature"] == 0.7
    assert a["gen_ai.request.max_tokens"] == 256
    assert a["gen_ai.request.top_p"] == 0.9
    assert a["llm.model_name"] == "openai/gpt-4o-mini"
    # invocation params exclude api_key
    assert "secret" not in a["llm.invocation_parameters"]
    assert "api_key" not in a["llm.invocation_parameters"]


def test_lm_request_attributes_minimal():
    a = semconv.lm_request_attributes(SimpleNamespace(model=None, kwargs={}))
    assert a["openinference.span.kind"] == "LLM"
    assert "gen_ai.request.model" not in a
    assert "gen_ai.system" not in a
    assert "llm.invocation_parameters" not in a


def test_lm_response_attributes_with_alias_usage_and_cost():
    choice = SimpleNamespace(finish_reason="length")
    entry = {
        "response_model": "gpt-4o-mini-2024",
        "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        "cost": 0.0003,
        "response": SimpleNamespace(choices=[choice]),
    }
    a = semconv.lm_response_attributes(entry)
    assert a["gen_ai.response.model"] == "gpt-4o-mini-2024"
    assert a["gen_ai.usage.input_tokens"] == 12
    assert a["llm.token_count.prompt"] == 12
    assert a["gen_ai.usage.output_tokens"] == 5
    assert a["gen_ai.usage.total_tokens"] == 17
    assert a["gen_ai.response.finish_reasons"] == ["length"]
    assert a["gen_ai.usage.cost"] == 0.0003


def test_lm_response_attributes_empty_entry():
    assert semconv.lm_response_attributes({}) == {}


def test_finish_reasons_bad_response_is_ignored():
    a = semconv.lm_response_attributes({"response": object()})
    assert "gen_ai.response.finish_reasons" not in a


def test_module_and_tool_kinds():
    assert semconv.module_attributes()["openinference.span.kind"] == "CHAIN"
    assert semconv.tool_attributes()["openinference.span.kind"] == "TOOL"
