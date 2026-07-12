"""Error classification and retry policy (all offline, synthetic exceptions)."""

import pytest

from support_crew import errors
from support_crew.errors import CrewRunError, ErrorKind


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class APIConnectionError(Exception):
    pass


def test_classify_auth_by_type_name():
    assert errors.classify(AuthenticationError("nope")) is ErrorKind.AUTH


def test_classify_auth_by_message():
    assert errors.classify(Exception("Incorrect API key provided")) is ErrorKind.AUTH


def test_classify_rate_limit():
    assert errors.classify(RateLimitError("slow down")) is ErrorKind.RATE_LIMIT
    assert errors.classify(Exception("HTTP 429")) is ErrorKind.RATE_LIMIT


def test_classify_timeout_and_network():
    assert errors.classify(TimeoutError("request timed out")) is ErrorKind.TIMEOUT
    assert errors.classify(APIConnectionError("boom")) is ErrorKind.NETWORK


def test_classify_walks_the_cause_chain():
    try:
        try:
            raise RateLimitError("inner")
        except RateLimitError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        assert errors.classify(outer) is ErrorKind.RATE_LIMIT


def test_classify_unknown_for_unrelated_errors():
    assert errors.classify(ValueError("bad value")) is ErrorKind.UNKNOWN


def test_user_messages_are_specific():
    auth = errors.message_for(ErrorKind.AUTH, AuthenticationError("x"))
    unknown = errors.message_for(ErrorKind.UNKNOWN, ValueError("details here"))
    assert "OPENAI_API_KEY" in auth
    assert "details here" in unknown


def test_retries_transient_error_with_backoff_then_succeeds():
    calls, delays = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise APIConnectionError("blip")
        return "ok"

    result = errors.execute_with_retries(
        flaky, attempts=3, base_delay=2.0, sleep=delays.append
    )
    assert result == "ok"
    assert len(calls) == 3
    assert delays == [2.0, 4.0]  # exponential backoff


def test_auth_error_fails_fast_without_retry():
    calls = []

    def bad_key():
        calls.append(1)
        raise AuthenticationError("invalid api key")

    with pytest.raises(CrewRunError) as excinfo:
        errors.execute_with_retries(
            bad_key, attempts=3, base_delay=0.0, sleep=lambda _: None
        )
    assert excinfo.value.kind is ErrorKind.AUTH
    assert len(calls) == 1


def test_exhausted_retries_raise_classified_error():
    def always_timeout():
        raise TimeoutError("timed out")

    with pytest.raises(CrewRunError) as excinfo:
        errors.execute_with_retries(
            always_timeout, attempts=2, base_delay=0.0, sleep=lambda _: None
        )
    assert excinfo.value.kind is ErrorKind.TIMEOUT


def test_abort_retry_vetoes_a_transient_retry():
    calls = []

    def flaky():
        calls.append(1)
        exc = APIConnectionError("blip")
        exc.record_written = True  # the failed attempt already saved
        raise exc

    with pytest.raises(CrewRunError):
        errors.execute_with_retries(
            flaky, attempts=3, base_delay=0.0,
            abort_retry=lambda exc: bool(getattr(exc, "record_written", False)),
            sleep=lambda _: None,
        )
    assert len(calls) == 1  # record already written -> never rerun


def test_run_with_deadline_returns_fast_results():
    assert errors.run_with_deadline(lambda: 42, timeout=5.0) == 42


def test_run_with_deadline_raises_non_retryable_timeout():
    import time as _time

    with pytest.raises(CrewRunError) as excinfo:
        errors.run_with_deadline(lambda: _time.sleep(2), timeout=0.05)
    assert excinfo.value.kind is ErrorKind.TIMEOUT


def test_deadline_breach_is_never_retried():
    """CrewRunError from the deadline must bypass the retry loop entirely."""
    calls = []

    def breaches_deadline():
        calls.append(1)
        raise CrewRunError(ErrorKind.TIMEOUT, "deadline")

    with pytest.raises(CrewRunError):
        errors.execute_with_retries(
            breaches_deadline, attempts=3, base_delay=0.0, sleep=lambda _: None
        )
    assert len(calls) == 1
