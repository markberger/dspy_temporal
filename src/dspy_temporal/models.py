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
    """Output of a program run: serialized Prediction + usage.

    Shared by both modes -- coarse runs the whole program in one activity, fine
    orchestrates it in the workflow; both return the same shape.
    """

    prediction: dict[str, Any]
    lm_usage: dict[str, Any] | None = None


# --- Fine-mode wire models ---------------------------------------------------
# These cross the per-call activity boundary, so (like everything in this
# module) they stay dspy-free and JSON-native. The workflow builds the input,
# the activity runs the real DSPy I/O and returns the output.


class LMCallInput(BaseModel):
    """One LM call delegated from the workflow to ``dspy_lm_call``."""

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    # JSON-safe sampling kwargs (temperature, max_tokens, ...). The workflow
    # strips anything non-serializable before sending (see ``serde.json_safe``).
    lm_kwargs: dict[str, Any] = Field(default_factory=dict)


class LMCallOutput(BaseModel):
    """Result of ``dspy_lm_call`` -- already-processed LM outputs + accounting."""

    # What ``dspy.LM.__call__`` returns: one entry per choice, each a parsed
    # string (ChatAdapter) or a dict (logprobs/tool_calls). The adapter parses
    # these back into fields workflow-side.
    outputs: list[Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    # The worker LM's request model id (e.g. ``openai/gpt-4o-mini``). Used as the
    # usage-tracker key so fine-mode ``lm_usage`` matches coarse mode exactly.
    model: str | None = None
    # The provider's response model id (e.g. ``gpt-4o-mini-2024-07-18``); carried
    # for parity/debuggability. The per-call span (emitted in the activity) reads
    # this straight from the LM history, not from here.
    response_model: str | None = None


class ToolCallInput(BaseModel):
    """One tool call delegated from the workflow to ``dspy_tool_call``."""

    program: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolCallOutput(BaseModel):
    """Result of ``dspy_tool_call`` -- the tool's JSON-safe observation."""

    observation: Any = None
