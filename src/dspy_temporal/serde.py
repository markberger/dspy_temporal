"""Serialization helpers for moving DSPy values across the Temporal boundary.

DSPy ``Prediction`` objects are backed by a plain ``_store`` dict (see
``dspy/primitives/prediction.py``), but the values may be pydantic models or
other rich types. Temporal payloads must be JSON-native, so we normalize the
store into JSON-safe primitives on the way out and reconstruct a ``Prediction``
on the way back in.
"""

from __future__ import annotations

from typing import Any

from dspy.primitives.prediction import Prediction
from pydantic import BaseModel


def _jsonify(value: Any) -> Any:
    """Best-effort conversion of an arbitrary value into JSON-native data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Prediction):
        return prediction_to_dict(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "model_dump"):  # pydantic-like without subclassing BaseModel
        try:
            return value.model_dump(mode="json")
        except Exception:  # pragma: no cover - defensive
            pass
    # Fallback: stringify so serialization never hard-fails on an exotic type.
    return str(value)


def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    """Convert a ``Prediction`` into a JSON-safe dict of its output fields."""
    store = prediction.toDict() if hasattr(prediction, "toDict") else dict(prediction._store)
    return {str(k): _jsonify(v) for k, v in store.items()}


def dict_to_prediction(data: dict[str, Any]) -> Prediction:
    """Reconstruct a ``Prediction`` from a previously serialized dict."""
    return Prediction(**data)


def normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Normalize program call inputs into JSON-safe data."""
    return {str(k): _jsonify(v) for k, v in inputs.items()}
