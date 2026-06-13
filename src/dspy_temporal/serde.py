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

    Used for tool args and for filtering an LM's ``kwargs`` into an ``LMSpec``.
    Values that JSON can't represent are *dropped* (not stringified) so an
    unsupported value degrades to its default rather than corrupting the call.
    (For LM *sampling* kwargs on the call path, use ``encode_lm_kwargs`` instead,
    which additionally carries a structured ``response_format`` across.)
    """
    safe: dict[str, Any] = {}
    for key, value in kwargs.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe[str(key)] = value
    return safe


# Marker key for a structured ``response_format`` (a pydantic model *class*) that
# can't be JSON-serialized directly. ``encode_lm_kwargs`` replaces the class with
# this marker carrying its JSON schema; ``decode_lm_kwargs`` rebuilds the
# litellm/OpenAI ``json_schema`` dict the worker LM accepts.
_RESPONSE_FORMAT_MARKER = "__dspy_temporal_response_format__"


def encode_lm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Encode LM sampling kwargs for the activity boundary (workflow side).

    Like ``json_safe`` (non-JSON values are dropped), but a structured
    ``response_format`` -- the pydantic model *class* ``JSONAdapter`` builds from
    the signature -- is preserved as a marker carrying its JSON schema instead of
    being dropped. ``{"type": "json_object"}`` and primitive kwargs pass through
    untouched. This is what lets ``JSONAdapter``/structured outputs cross into the
    activity (the class itself isn't JSON-serializable).
    """
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if (
            key == "response_format"
            and isinstance(value, type)
            and issubclass(value, BaseModel)
        ):
            out[key] = {
                _RESPONSE_FORMAT_MARKER: {
                    "name": value.__name__,
                    "json_schema": value.model_json_schema(),
                }
            }
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        out[str(key)] = value
    return out


def decode_lm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Decode encoded LM sampling kwargs in the activity (worker side).

    Turns a ``response_format`` marker back into the litellm/OpenAI
    ``{"type": "json_schema", "json_schema": {...}}`` form -- which litellm
    accepts directly -- so we avoid reconstructing a pydantic class. Nothing
    downstream of the LM call needs the class: ``JSONAdapter.parse`` (run in the
    workflow) uses only the signature's output fields.
    """
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if (
            key == "response_format"
            and isinstance(value, dict)
            and _RESPONSE_FORMAT_MARKER in value
        ):
            marker = value[_RESPONSE_FORMAT_MARKER]
            out[key] = {
                "type": "json_schema",
                "json_schema": {
                    "name": marker["name"],
                    "schema": marker["json_schema"],
                    "strict": True,
                },
            }
        else:
            out[key] = value
    return out
