"""Serde round-trip tests."""

import json

import dspy
from pydantic import BaseModel

from dspy_temporal.serde import (
    _jsonify,
    decode_lm_kwargs,
    dict_to_prediction,
    encode_lm_kwargs,
    json_safe,
    normalize_inputs,
    prediction_to_dict,
)


class _Meta(BaseModel):
    score: int


def test_prediction_roundtrip_is_json_safe():
    pred = dspy.Prediction(
        answer="42",
        reasoning="because",
        meta=_Meta(score=7),
        tags=["a", "b"],
    )
    data = prediction_to_dict(pred)

    # Must be JSON-serializable (Temporal payloads are JSON-native).
    json.dumps(data)

    assert data["answer"] == "42"
    assert data["meta"] == {"score": 7}
    assert data["tags"] == ["a", "b"]

    restored = dict_to_prediction(data)
    assert restored.answer == "42"
    assert restored.meta == {"score": 7}


def test_normalize_inputs_stringifies_keys_and_jsonifies_values():
    out = normalize_inputs({"question": "hi", "ctx": _Meta(score=1)})
    json.dumps(out)
    assert out["question"] == "hi"
    assert out["ctx"] == {"score": 1}


def test_exotic_value_falls_back_to_str():
    class Weird:
        def __str__(self):
            return "weird!"

    data = prediction_to_dict(dspy.Prediction(x=Weird()))
    json.dumps(data)
    assert data["x"] == "weird!"


def test_nested_prediction_is_recursed():
    inner = dspy.Prediction(answer="42")
    data = prediction_to_dict(dspy.Prediction(result=inner, tup=(1, 2)))
    json.dumps(data)
    assert data["result"] == {"answer": "42"}
    assert data["tup"] == [1, 2]  # tuple -> list


def test_pydantic_like_without_basemodel_uses_model_dump():
    class Faux:
        def model_dump(self, mode=None):
            return {"dumped": True, "mode": mode}

    data = prediction_to_dict(dspy.Prediction(obj=Faux()))
    json.dumps(data)
    assert data["obj"] == {"dumped": True, "mode": "json"}


def test_normalize_inputs_stringifies_non_str_keys():
    out = normalize_inputs({1: "x"})
    assert out == {"1": "x"}


def test_jsonify_handles_a_raw_prediction():
    # _jsonify is a reusable helper; it must convert a Prediction directly even
    # though prediction_to_dict's toDict() normally pre-flattens nested ones.
    assert _jsonify(dspy.Prediction(answer="42")) == {"answer": "42"}


def test_json_safe_keeps_primitives_and_stringifies_keys():
    # The ChatAdapter case: every LM sampling kwarg is a JSON primitive, so all
    # survive (and keys are coerced to str).
    out = json_safe({"temperature": 0.7, "max_tokens": 256, 1: "x"})
    json.dumps(out)
    assert out == {"temperature": 0.7, "max_tokens": 256, "1": "x"}


def test_json_safe_drops_non_serializable_values():
    # json_safe backs tool args and the LMSpec.kwargs filter: a value JSON can't
    # represent (here a pydantic *class*) is dropped rather than corrupting the
    # payload. (LM sampling kwargs use encode_lm_kwargs, which instead *carries* a
    # structured response_format across -- see below.)
    out = json_safe({"temperature": 0.0, "response_format": _Meta})
    json.dumps(out)
    assert out == {"temperature": 0.0}
    assert "response_format" not in out


# --- LM-kwargs codec (encode_lm_kwargs / decode_lm_kwargs) -------------------


def test_encode_lm_kwargs_passes_through_primitives_and_json_object():
    # ChatAdapter primitives and the JSONAdapter json_object fallback are already
    # JSON-native, so they cross untouched.
    out = encode_lm_kwargs(
        {"temperature": 0.0, "response_format": {"type": "json_object"}}
    )
    json.dumps(out)
    assert out == {"temperature": 0.0, "response_format": {"type": "json_object"}}


def test_encode_lm_kwargs_carries_structured_response_format():
    # The JSONAdapter structured-output case: the pydantic response_format *class*
    # is now carried (as a schema marker) instead of dropped.
    out = encode_lm_kwargs({"response_format": _Meta})
    json.dumps(out)  # JSON-native now
    marker = out["response_format"]["__dspy_temporal_response_format__"]
    assert marker["name"] == "_Meta"
    assert marker["json_schema"] == _Meta.model_json_schema()


def test_encode_lm_kwargs_drops_other_non_json():
    out = encode_lm_kwargs({"temperature": 0.0, "weird": lambda: 1})  # noqa: E731
    assert out == {"temperature": 0.0}


def test_decode_lm_kwargs_rebuilds_litellm_json_schema():
    decoded = decode_lm_kwargs(encode_lm_kwargs({"response_format": _Meta}))
    rf = decoded["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Meta"
    assert rf["json_schema"]["schema"] == _Meta.model_json_schema()
    assert rf["json_schema"]["strict"] is True


def test_decode_lm_kwargs_passes_through_plain_values():
    decoded = decode_lm_kwargs(
        {"temperature": 0.0, "response_format": {"type": "json_object"}}
    )
    assert decoded == {"temperature": 0.0, "response_format": {"type": "json_object"}}
