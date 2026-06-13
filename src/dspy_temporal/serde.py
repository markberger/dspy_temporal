"""Serialization helpers for moving DSPy values across the Temporal boundary.

DSPy ``Prediction`` objects are backed by a plain ``_store`` dict (see
``dspy/primitives/prediction.py``), but the values may be pydantic models or
other rich types. Temporal payloads must be JSON-native, so we normalize the
store into JSON-safe primitives on the way out and reconstruct a ``Prediction``
on the way back in.
"""

from __future__ import annotations

import json
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


def dict_to_prediction(
    data: dict[str, Any], lm_usage: dict[str, Any] | None = None
) -> Prediction:
    """Reconstruct a ``Prediction`` from a previously serialized dict.

    ``lm_usage`` (the per-LM token totals carried alongside the prediction in
    ``ProgramCallOutput``) is restored via ``set_lm_usage`` when present. It
    lives *outside* ``_store`` -- a separate ``Prediction`` attribute -- so it is
    not part of ``data`` and would otherwise be lost across the Temporal
    boundary, leaving ``get_lm_usage()`` empty on the returned prediction.
    """
    prediction = Prediction(**data)
    if lm_usage:
        prediction.set_lm_usage(lm_usage)
    return prediction


def normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Normalize program call inputs into JSON-safe data."""
    return {str(k): _jsonify(v) for k, v in inputs.items()}


def json_safe(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the entries that round-trip cleanly through JSON.

    Fine mode ships LM sampling kwargs and tool args across the activity
    boundary. Values that JSON can't represent are *dropped* (not stringified)
    so an unsupported kwarg degrades to its default rather than corrupting the
    call -- e.g. ``JSONAdapter``'s pydantic ``response_format`` class is omitted
    (documented limitation) instead of being coerced into a bogus string. For
    the default ``ChatAdapter`` every kwarg is a primitive, so nothing is lost.
    """
    safe: dict[str, Any] = {}
    for key, value in kwargs.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe[str(key)] = value
    return safe
