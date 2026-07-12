"""Error taxonomy and retry policy for crew runs.

Provider exceptions (litellm / openai / httpx) are classified by NAME and
MESSAGE rather than by importing their exception classes: those internals
move between releases, while the names ("AuthenticationError",
"RateLimitError", …) are stable API surface.
"""

import logging
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorKind(Enum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    UNKNOWN = "unknown"


#: Kinds worth retrying — the next attempt can plausibly succeed unchanged.
TRANSIENT_KINDS = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.NETWORK})

# Ordered: first match wins. AUTH is checked before NETWORK because auth
# failures often mention the HTTP connection too.
_PATTERNS: tuple[tuple[ErrorKind, tuple[str, ...]], ...] = (
    (
        ErrorKind.AUTH,
        (
            "authenticationerror", "permissiondenied", "unauthorized",
            "invalid api key", "incorrect api key", "401", "403",
        ),
    ),
    (ErrorKind.RATE_LIMIT, ("ratelimit", "quota", "429")),
    (ErrorKind.TIMEOUT, ("timeout", "timed out", "deadline")),
    (
        ErrorKind.NETWORK,
        (
            "connectionerror", "apiconnection", "serviceunavailable",
            "badgateway", "connection refused", "connection reset",
            "temporarily unavailable", "500", "502", "503",
        ),
    ),
)

_USER_MESSAGES = {
    ErrorKind.AUTH: (
        "Authentication with the AI services failed. Check that your "
        "OPENAI_API_KEY and SERPER_API_KEY are valid, then try again."
    ),
    ErrorKind.RATE_LIMIT: (
        "The AI service is rate-limiting requests right now. "
        "Wait a moment and try again."
    ),
    ErrorKind.TIMEOUT: (
        "The agents took too long and the run was stopped. "
        "Try again — a shorter, more specific query may help."
    ),
    ErrorKind.NETWORK: (
        "A network problem interrupted the run. Check your internet "
        "connection and try again."
    ),
}


class CrewRunError(RuntimeError):
    """A crew run failed; carries a classified kind and a user-safe message."""

    def __init__(self, kind: ErrorKind, user_message: str) -> None:
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message


def _texts(exc: BaseException) -> Iterator[str]:
    """Type names and messages of the exception and everything it wraps."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield type(current).__name__
        yield str(current)
        current = current.__cause__ or current.__context__


def classify(exc: BaseException) -> ErrorKind:
    """Map an arbitrary exception (and its cause chain) to an ErrorKind."""
    haystack = " ".join(_texts(exc)).lower()
    for kind, needles in _PATTERNS:
        if any(needle in haystack for needle in needles):
            return kind
    return ErrorKind.UNKNOWN


def message_for(kind: ErrorKind, exc: BaseException) -> str:
    """User-facing message for a classified failure (never leaks secrets)."""
    if kind in _USER_MESSAGES:
        return _USER_MESSAGES[kind]
    return (
        "Something went wrong while running the crew: "
        f"{type(exc).__name__}: {exc}"
    )


def execute_with_retries[T](
    operation: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    abort_retry: Callable[[BaseException], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `operation`, retrying transient failures with exponential backoff.

    Non-transient failures (auth, unknown) raise immediately, as does any
    CrewRunError the operation classified itself. `abort_retry` receives
    the exception and lets the caller veto a retry that is no longer safe —
    e.g. when the failed attempt already wrote its record, a rerun would
    duplicate it. Every failure surfaces as CrewRunError with a user-safe
    message.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except CrewRunError:
            raise
        except Exception as exc:
            kind = classify(exc)
            may_retry = (
                kind in TRANSIENT_KINDS
                and attempt < attempts
                and not (abort_retry(exc) if abort_retry is not None else False)
            )
            if not may_retry:
                raise CrewRunError(kind, message_for(kind, exc)) from exc
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "Transient %s error on attempt %d/%d (%s); retrying in %.0fs",
                kind.value, attempt, attempts, type(exc).__name__, delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # loop always returns or raises


def run_with_deadline[T](operation: Callable[[], T], timeout: float) -> T:
    """Run `operation` in a worker thread with a hard overall deadline.

    Guards the caller (a Streamlit session thread) against a run that hangs
    past every per-agent timeout — the UI gets a clean TIMEOUT error instead
    of spinning forever. Python threads cannot be killed, so on deadline the
    worker is abandoned (it usually dies shortly after via the per-agent
    `max_execution_time`). The breach is raised as a non-retryable
    CrewRunError: the abandoned worker could still write its record, so an
    automatic rerun might duplicate it.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crew-run")
    try:
        future = pool.submit(operation)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            logger.warning("Run exceeded the %.0fs deadline; abandoning worker", timeout)
            raise CrewRunError(
                ErrorKind.TIMEOUT, _USER_MESSAGES[ErrorKind.TIMEOUT]
            ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
