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


class LMSpec(BaseModel):
    """A predictor's *effective* LM, described on the worker for the workflow.

    The fine workflow needs each predictor's model id, capability flags, and
    sampling kwargs *before* it dispatches the first LM call: ``JSONAdapter``
    branches on ``supports_response_schema`` / ``supported_params`` and
    ``Predict._forward_preprocess`` derives config from ``lm.kwargs`` -- both in
    the workflow. ``WorkflowLM`` carries this spec so those decisions match the
    real worker LM. All fields are JSON-native (no credentials: ``kwargs`` is
    filtered worker-side; api keys are read from env at call time in the activity).
    """

    model: str
    model_type: str = "chat"
    supported_params: list[str] = Field(default_factory=list)
    supports_response_schema: bool = False
    supports_function_calling: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)


class LMDescribeInput(BaseModel):
    """Input to ``dspy_describe_lms``: which program to introspect."""

    program: str


class LMSpecsOutput(BaseModel):
    """Per-predictor ``LMSpec`` map (plus a ``"__default__"`` worker-LM entry)."""

    specs: dict[str, LMSpec] = Field(default_factory=dict)


class LMCallInput(BaseModel):
    """One LM call delegated from the workflow to ``dspy_lm_call``."""

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    # Sampling kwargs (temperature, max_tokens, response_format, ...) encoded for
    # the wire by ``serde.encode_lm_kwargs`` and decoded in the activity.
    lm_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Which predictor's LM to run (a name from ``named_predictors()``); None or
    # ``"__default__"`` -> the worker default LM. Lets the activity honor a
    # per-predictor bound ``.lm`` without sending any LM/credentials over the wire.
    lm_ref: str | None = None
    # The program name, so the activity can resolve ``lm_ref`` -> a real LM via
    # the registry (cached). None keeps the old worker-default behavior.
    program: str | None = None


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
