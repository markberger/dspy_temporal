"""Serializable activity options carried in the workflow input.

This module is intentionally free of any ``dspy`` import so it can be imported
inside the workflow sandbox without dragging in litellm/http machinery.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum

from pydantic import BaseModel, Field
from temporalio.common import RetryPolicy

# Errors that should never be retried: a bigger prompt won't fit on retry, and a
# parse error is deterministic given the same model output.
DEFAULT_NON_RETRYABLE = ["ContextWindowExceededError"]


class RunMode(str, Enum):
    """How a deployed program runs on Temporal.

    - ``COARSE``: the whole ``dspy.Module`` runs in one activity; durability is
      job-level (a crash re-runs the whole program).
    - ``FINE``: each LM call and each tool call is its own activity, orchestrated
      by the workflow, so completed steps survive a crash and are not re-run.

    A ``str`` mix-in (rather than 3.11+ ``enum.StrEnum``, since we support 3.10)
    keeps the members JSON/repr-friendly.
    """

    COARSE = "coarse"
    FINE = "fine"


class CallOptions(BaseModel):
    """Timeouts + retry settings for the program activity.

    Temporal's ``RetryPolicy``/timeouts are not passed as payloads directly; we
    send these primitives and rebuild the policy inside the workflow.
    """

    start_to_close_timeout_seconds: float = 300.0
    # Set well above your slowest single LM/tool call and above ~1s: when set, the
    # activity self-heartbeats at ~1/3 this interval (see heartbeat.py), so a
    # configured value keeps the activity alive instead of guaranteeing a
    # mid-flight HEARTBEAT timeout. None (the default) disables heartbeating.
    heartbeat_timeout_seconds: float | None = None
    maximum_attempts: int = 3
    initial_interval_seconds: float = 1.0
    backoff_coefficient: float = 2.0
    maximum_interval_seconds: float = 60.0
    non_retryable_error_types: list[str] = Field(
        default_factory=lambda: list(DEFAULT_NON_RETRYABLE)
    )

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

    def activity_kwargs(self, *, task_queue: str | None = None) -> dict:
        """Timeouts/retry/heartbeat kwargs for ``workflow.execute_activity``.

        The single source of truth shared by every activity dispatch (coarse and
        fine) so the option set behaves identically across modes. A ``None``
        ``heartbeat_timeout`` is equivalent to omitting it (no heartbeat timeout).

        ``task_queue`` routes the activity to a dedicated queue (the cheap-workflow-
        workers + dedicated-activity-pool split); ``None`` co-locates it with the
        calling workflow's queue. Temporal rejects an explicit ``task_queue=None``,
        so it is only added to the kwargs when set.
        """
        kwargs = {
            "start_to_close_timeout": self.start_to_close_timeout(),
            "heartbeat_timeout": self.heartbeat_timeout(),
            "retry_policy": self.retry_policy(),
        }
        if task_queue is not None:
            kwargs["task_queue"] = task_queue
        return kwargs
