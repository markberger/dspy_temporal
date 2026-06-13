"""Serde round-trip tests."""

import json

import dspy
from pydantic import BaseModel

from dspy_temporal.serde import (
    _jsonify,
    dict_to_prediction,
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
    # The JSONAdapter case (documented fine-mode limitation): a pydantic
    # response_format *class* can't cross the activity boundary, so it is dropped
    # (degrades to the default) rather than corrupting the call.
    out = json_safe({"temperature": 0.0, "response_format": _Meta})
    json.dumps(out)
    assert out == {"temperature": 0.0}
    assert "response_format" not in out
