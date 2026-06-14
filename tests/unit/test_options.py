"""Tests for CallOptions -> timeouts / RetryPolicy mapping."""

from datetime import timedelta

from dspy_temporal.options import DEFAULT_NON_RETRYABLE, CallOptions


def test_defaults():
    o = CallOptions()
    assert o.start_to_close_timeout_seconds == 300.0
    assert o.heartbeat_timeout_seconds is None
    assert o.maximum_attempts == 3
    assert o.backoff_coefficient == 2.0
    assert o.non_retryable_error_types == DEFAULT_NON_RETRYABLE
    assert "ContextWindowExceededError" in o.non_retryable_error_types


def test_default_factory_lists_are_independent():
    a, b = CallOptions(), CallOptions()
    a.non_retryable_error_types.append("Other")
    assert "Other" not in b.non_retryable_error_types


def test_start_to_close_timeout_conversion():
    assert CallOptions(
        start_to_close_timeout_seconds=12
    ).start_to_close_timeout() == timedelta(seconds=12)


def test_heartbeat_timeout_none_and_value():
    assert CallOptions().heartbeat_timeout() is None
    assert CallOptions(heartbeat_timeout_seconds=5).heartbeat_timeout() == timedelta(
        seconds=5
    )


def test_retry_policy_maps_all_fields():
    o = CallOptions(
        maximum_attempts=7,
        initial_interval_seconds=2,
        backoff_coefficient=3.0,
        maximum_interval_seconds=90,
        non_retryable_error_types=["FooError"],
    )
    rp = o.retry_policy()
    assert rp.maximum_attempts == 7
    assert rp.initial_interval == timedelta(seconds=2)
    assert rp.backoff_coefficient == 3.0
    assert rp.maximum_interval == timedelta(seconds=90)
    assert rp.non_retryable_error_types == ["FooError"]


def test_retry_policy_non_retryable_is_a_copy():
    o = CallOptions(non_retryable_error_types=["FooError"])
    rp = o.retry_policy()
    rp.non_retryable_error_types.append("BarError")
    assert o.non_retryable_error_types == ["FooError"]  # source untouched


def test_activity_kwargs_keys_and_default_heartbeat():
    kw = CallOptions().activity_kwargs()
    assert set(kw) == {"start_to_close_timeout", "heartbeat_timeout", "retry_policy"}
    assert kw["start_to_close_timeout"] == timedelta(seconds=300)
    assert kw["heartbeat_timeout"] is None  # default: equivalent to omitting it
    assert kw["retry_policy"].maximum_attempts == 3


def test_activity_kwargs_includes_heartbeat_when_set():
    kw = CallOptions(heartbeat_timeout_seconds=5).activity_kwargs()
    assert kw["heartbeat_timeout"] == timedelta(seconds=5)
