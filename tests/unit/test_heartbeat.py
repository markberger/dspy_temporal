"""Tests for the self-heartbeating watchdog (no Temporal server).

``ActivityEnvironment`` runs a callable inside an activity context: its ``info``
(customizable via ``dataclasses.replace``) carries the ``heartbeat_timeout`` the
watchdog self-discovers, and its ``on_heartbeat`` callback fires on every
``activity.heartbeat()``. That lets us prove the watchdog beats when configured,
no-ops when not, and tears its thread down -- deterministically, no fixed sleeps.
"""

import dataclasses
import threading
from datetime import timedelta

from temporalio import activity
from temporalio.testing import ActivityEnvironment

from dspy_temporal.heartbeat import heartbeating

_THREAD_NAME = "dspy-activity-heartbeat"


def _env(heartbeat_timeout: timedelta | None) -> ActivityEnvironment:
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_timeout=heartbeat_timeout)
    return env


def test_noop_when_no_heartbeat_timeout():
    """Default (no heartbeat_timeout): no beats, no watchdog thread spawned."""
    env = _env(None)
    beats = []
    env.on_heartbeat = lambda *a: beats.append(a)

    def body():
        with heartbeating():
            assert not any(t.name == _THREAD_NAME for t in threading.enumerate())

    env.run(body)
    assert beats == []


def test_beats_during_blocking_body():
    """With a heartbeat_timeout set, the watchdog beats while the body blocks."""
    env = _env(timedelta(seconds=0.3))  # interval = 0.3/3 = 0.1s
    beat_seen = threading.Event()
    env.on_heartbeat = lambda *a: beat_seen.set()

    def body():
        with heartbeating():
            # Unblocks the moment the first beat lands (or fails after 5s) -- no
            # fixed sleep, so the test is fast and not timing-flaky.
            assert beat_seen.wait(timeout=5.0), "no heartbeat fired within 5s"

    env.run(body)
    assert beat_seen.is_set()


def test_watchdog_thread_stops_after_block():
    """The bounded join leaves no live watchdog thread once the block exits."""
    env = _env(timedelta(seconds=0.3))
    beat_seen = threading.Event()
    env.on_heartbeat = lambda *a: beat_seen.set()

    def body():
        with heartbeating():
            beat_seen.wait(timeout=5.0)

    env.run(body)
    assert not any(
        t.name == _THREAD_NAME and t.is_alive() for t in threading.enumerate()
    )


def test_heartbeat_errors_are_non_fatal():
    """A raising heartbeat (e.g. loop closing at shutdown) doesn't kill the run."""
    env = _env(timedelta(seconds=0.3))
    raised = threading.Event()

    def boom(*_a):
        raised.set()
        raise RuntimeError("heartbeat backend gone")

    env.on_heartbeat = boom

    def body():
        with heartbeating():
            assert raised.wait(timeout=5.0), "watchdog never attempted a heartbeat"

    # Must not propagate the watchdog's exception.
    env.run(body)
    assert raised.is_set()


def test_heartbeating_inside_real_activity_defn():
    """activity.info() resolves inside a decorated activity, so the wrap is live."""
    env = _env(timedelta(seconds=0.3))
    beat_seen = threading.Event()
    env.on_heartbeat = lambda *a: beat_seen.set()

    @activity.defn
    def act() -> str:
        with heartbeating():
            beat_seen.wait(timeout=5.0)
        return "done"

    assert env.run(act) == "done"
    assert beat_seen.is_set()
