"""Serializable activity options carried in the workflow input.

This module is intentionally free of any ``dspy`` import so it can be imported
inside the workflow sandbox without dragging in litellm/http machinery.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field
from temporalio.common import RetryPolicy

# Errors that should never be retried: a bigger prompt won't fit on retry, and a
# parse error is deterministic given the same model output.
DEFAULT_NON_RETRYABLE = ["ContextWindowExceededError"]


class CallOptions(BaseModel):
    """Timeouts + retry settings for the program activity.

    Temporal's ``RetryPolicy``/timeouts are not passed as payloads directly; we
    send these primitives and rebuild the policy inside the workflow.
    """

    start_to_close_timeout_seconds: float = 300.0
    heartbeat_timeout_seconds: float | None = None
    maximum_attempts: int = 3
    initial_interval_seconds: float = 1.0
    backoff_coefficient: float = 2.0
    maximum_interval_seconds: float = 60.0
    non_retryable_error_types: list[str] = Field(default_factory=lambda: list(DEFAULT_NON_RETRYABLE))

    def start_to_close_timeout(self) -> timedelta:
        return timedelta(seconds=self.start_to_close_timeout_seconds)

    def heartbeat_timeout(self) -> timedelta | None:
        if self.heartbeat_timeout_seconds is None:
            return None
        return timedelta(seconds=self.heartbeat_timeout_seconds)

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            initial_interval=timedelta(seconds=self.initial_interval_seconds),
            backoff_coefficient=self.backoff_coefficient,
            maximum_interval=timedelta(seconds=self.maximum_interval_seconds),
            maximum_attempts=self.maximum_attempts,
            non_retryable_error_types=list(self.non_retryable_error_types),
        )
