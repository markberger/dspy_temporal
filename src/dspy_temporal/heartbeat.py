"""Self-heartbeating context manager for synchronous DSPy activities.

Temporal fails an activity with a HEARTBEAT timeout if nothing calls
``activity.heartbeat()`` within the configured ``heartbeat_timeout``. Our
activities are synchronous and block on a DSPy/LM/tool call that never yields, so
they cannot beat between operations -- a background daemon thread must supply the
liveness while the call runs.

Two subtleties make this non-trivial, both handled here:

- ``activity.heartbeat()`` resolves the activity context from a
  ``contextvars.ContextVar`` that a plain ``threading.Thread`` does NOT inherit,
  so we capture ``contextvars.copy_context()`` at activity entry (where the
  context IS set) and run the loop via ``ctx.run(...)``. The heartbeat callable
  itself is thread-safe (it marshals onto the worker event loop).
- The activity self-discovers its cadence from ``activity.info().heartbeat_timeout``,
  so no heartbeat value has to be threaded through the activity payload.

When no heartbeat timeout is configured (the default) this is a strict no-op: no
thread is spawned, so the common case pays nothing.

Activity-only module: it imports ``temporalio.activity`` + stdlib and must never
be imported by workflow/sandbox code.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from temporalio import activity

# Beat ~3x per timeout window so one slow/dropped beat doesn't trip the timeout.
_DIVISOR = 3
# The heartbeat accept call can block up to 10s; never spin the loop faster than
# this floor (a pathologically small heartbeat_timeout is user error).
_MIN_INTERVAL_SECONDS = 0.1


@contextmanager
def heartbeating() -> Iterator[None]:
    """Heartbeat from a background daemon thread for the duration of the block.

    No-op when the activity has no ``heartbeat_timeout`` configured. Otherwise
    beats every ``heartbeat_timeout / 3`` (floored at 0.1s) until the block exits.
    """
    timeout = activity.info().heartbeat_timeout
    if timeout is None:
        yield
        return

    interval = max(timeout.total_seconds() / _DIVISOR, _MIN_INTERVAL_SECONDS)
    stop = threading.Event()
    # Snapshot the activity context here, on the activity's worker thread, so the
    # watchdog thread can run activity.heartbeat() within it.
    ctx = contextvars.copy_context()

    def _loop() -> None:
        # stop.wait returns True once set (-> exit) or False on timeout (-> beat).
        while not stop.wait(interval):
            try:
                activity.heartbeat()
            except Exception:  # noqa: S112 - non-fatal: a failed beat (e.g. the worker
                # loop closing at shutdown) must not crash this daemon thread or
                # gate the activity; the next iteration retries or stop ends it.
                continue

    thread = threading.Thread(
        target=lambda: ctx.run(_loop),
        name="dspy-activity-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        # Bounded: don't let an in-flight beat's (up to 10s) accept wait gate the
        # activity's return. The daemon flag backstops any wedged beat.
        thread.join(timeout=1.0)
