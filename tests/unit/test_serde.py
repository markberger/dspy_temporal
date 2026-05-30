"""Serde round-trip tests."""

import json

import dspy
from pydantic import BaseModel

from dspy_temporal.serde import dict_to_prediction, normalize_inputs, prediction_to_dict


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
