"""Shared pydantic payload models for workflow/activity I/O."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .options import CallOptions


class ProgramCallInput(BaseModel):
    """Input to a coarse program workflow/activity."""

    program: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    options: CallOptions | None = None


class ProgramCallOutput(BaseModel):
    """Output of a coarse program run: serialized Prediction + usage."""

    prediction: dict[str, Any]
    lm_usage: dict[str, Any] | None = None
