"""
Multi-Agent Customer Support System
===================================
A CrewAI application with exactly three agents running sequentially:

    1. Assistant            -> answers the query from its own LLM knowledge
    2. Web Search Assistant -> answers the same query using Serper web search
    3. Entry Agent          -> saves the query + both answers to answers.txt

The crew is coordinated with `process=Process.sequential`, and Task 3
receives the outputs of Task 1 and Task 2 through CrewAI task `context`.

Per the assignment, ALL application code lives in this single file,
organised in sections:

    Configuration  -- settings, env loading, API-key validation
    Models         -- SupportRecord (structured output) + RunResult
    Errors         -- error taxonomy, retries with backoff, run deadline
    Storage        -- locked answers.txt appends, Record-IDs, rotation
    Tool           -- the Entry Agent's file-saving tool
    Crew           -- agents, tasks, sequential crew, execution + logging
    UI             -- Streamlit views, components, and styles

Run with:
    streamlit run app.py
"""

import base64
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from filelock import FileLock
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: every environmental assumption lives here.
#
# Settings resolve in precedence order:
#     1. shell environment variables,
#     2. a local .env file (NAME=value lines, loaded by python-dotenv),
#     3. Streamlit secrets (.streamlit/secrets.toml locally, or the Secrets
#        panel on Streamlit Community Cloud) — filled into the environment
#        only for names not already set.
#
# Key VALUES are never printed, logged, or displayed anywhere.
# ---------------------------------------------------------------------------

load_dotenv()

# Every name this app reads from the environment (used for the st.secrets
# fallback below and documented in .env.example).
_ENV_NAMES = (
    "OPENAI_API_KEY",
    "SERPER_API_KEY",
    "SUPPORT_CREW_MODEL",
    "SUPPORT_CREW_AGENT_TIMEOUT",
    "SUPPORT_CREW_RUN_TIMEOUT",
    "SUPPORT_CREW_MAX_QUERY_CHARS",
    "SUPPORT_CREW_MAX_ATTEMPTS",
    "SUPPORT_CREW_RETRY_BASE_DELAY",
    "SUPPORT_CREW_ANSWERS_MAX_BYTES",
)


def _fill_env_from_streamlit_secrets() -> None:
    """Copy known names from st.secrets into the environment (gaps only).

    Makes the app deployable on Streamlit Community Cloud, where secrets
    arrive via st.secrets rather than shell variables. Environment / .env
    values always win. Silently a no-op when no secrets file exists.
    """
    try:
        for name in _ENV_NAMES:
            if name not in os.environ and name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except Exception:  # noqa: BLE001 - missing secrets.toml is normal
        pass


_fill_env_from_streamlit_secrets()

PROJECT_ROOT = Path(__file__).resolve().parent

# answers.txt lives in the data/ folder (git-ignored); both the folder and
# the file are created automatically on first save.
ANSWERS_FILE = PROJECT_ROOT / "data" / "answers.txt"

# The LLM every agent runs on. Pinned explicitly so behaviour and cost do not
# drift with library defaults; override without a code change via env var.
MODEL_NAME = os.environ.get("SUPPORT_CREW_MODEL", "gpt-4.1-mini")

# Hard ceiling (seconds) on each agent's execution so a hung provider call
# can never spin the UI forever.
AGENT_MAX_EXECUTION_TIME = int(os.environ.get("SUPPORT_CREW_AGENT_TIMEOUT", "120"))

# Overall deadline (seconds) for one crew run — the safety net above the
# per-agent ceiling (3 agents x 120s + margin by default).
RUN_TIMEOUT = int(os.environ.get("SUPPORT_CREW_RUN_TIMEOUT", "420"))

# Maximum accepted query length (characters). Guards against accidental or
# malicious giant inputs turning into unbounded token spend.
MAX_QUERY_CHARS = int(os.environ.get("SUPPORT_CREW_MAX_QUERY_CHARS", "2000"))

# Retry policy for transient provider failures (rate limit, network,
# timeout). Non-transient failures (e.g. bad API key) never retry.
MAX_ATTEMPTS = int(os.environ.get("SUPPORT_CREW_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY = float(os.environ.get("SUPPORT_CREW_RETRY_BASE_DELAY", "2"))

# answers.txt is rotated to a timestamped archive once it reaches this size,
# so the file (which holds user queries) can never grow without bound.
# Set to 0 to disable rotation.
ANSWERS_MAX_BYTES = int(
    os.environ.get("SUPPORT_CREW_ANSWERS_MAX_BYTES", str(5 * 1024 * 1024))
)

# Environment variables that must be set before the crew can run.
REQUIRED_API_KEYS = ("OPENAI_API_KEY", "SERPER_API_KEY")


def missing_api_keys() -> list[str]:
    """Return the names of any required API-key environment variables not set."""
    return [name for name in REQUIRED_API_KEYS if not os.environ.get(name, "").strip()]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SupportRecord(BaseModel):
    """Reliably parseable final output returned by the Entry Agent."""

    assistant_answer: str
    web_search_answer: str
    file_saved: bool
    file_path: str


@dataclass
class RunResult:
    """Typed outcome of one crew run; stored in session state and history."""

    query: str
    assistant_answer: str
    web_search_answer: str
    file_saved: bool
    file_path: str
    completed_at: float | None = None
    # Token accounting (from CrewAI's usage metrics; None when unavailable).
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
# Errors: taxonomy and retry policy for crew runs.
#
# Provider exceptions (litellm / openai / httpx) are classified by NAME and
# MESSAGE rather than by importing their exception classes: those internals
# move between releases, while the names ("AuthenticationError",
# "RateLimitError", …) are stable API surface.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Storage: answers.txt persistence.
#
# All writes go through `append_record`, which:
#   * assigns every record a unique Record-ID,
#   * takes an OS-level file lock so concurrent sessions can never interleave
#     or corrupt records,
#   * remembers the last record ID written on this thread, so the caller that
#     triggered the crew run can verify *its* record landed (a plain
#     file-size delta cannot distinguish between two concurrent sessions).
# ---------------------------------------------------------------------------

RECORD_TEMPLATE = (
    "============================================================\n"
    "MULTI-AGENT CUSTOMER SUPPORT RESPONSE\n"
    "Record-ID: {record_id}\n"
    "============================================================\n"
    "\n"
    "Query:\n"
    "{query}\n"
    "\n"
    "------------------------------------------------------------\n"
    "Assistant Answer:\n"
    "{assistant_answer}\n"
    "\n"
    "------------------------------------------------------------\n"
    "Web Search Answer:\n"
    "{web_search_answer}\n"
    "\n"
    "============================================================\n"
    "\n"
)

# The Entry Agent's tool runs synchronously inside the same thread as the
# crew run that invoked it, so a thread-local is the right scope for
# "the record MY run just wrote".
_local = threading.local()


def _lock_for(path: Path) -> FileLock:
    """One sidecar .lock file per answers file (git-ignored)."""
    return FileLock(f"{path}.lock")


def _rotate_if_needed(path: Path, max_bytes: int) -> None:
    """Archive the answers file once it reaches the size cap.

    Rotation keeps the file (which accumulates user queries) from growing
    without bound. Must be called while holding the file lock. A cap of 0
    or less disables rotation.
    """
    if max_bytes <= 0 or not path.exists() or path.stat().st_size < max_bytes:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    if archive.exists():  # two rotations within one second
        archive = path.with_name(f"{path.stem}-{stamp}-{uuid.uuid4().hex[:8]}{path.suffix}")
    path.rename(archive)
    logger.info(
        "Rotated %s to %s (size cap %d bytes reached)",
        path.name, archive.name, max_bytes,
    )


def append_record(
    query: str,
    assistant_answer: str,
    web_search_answer: str,
    *,
    path: Path | None = None,
    max_bytes: int | None = None,
) -> str:
    """Append one support record under a file lock; return its Record-ID.

    Append mode preserves earlier records; UTF-8 is explicit. The generated
    Record-ID is also remembered thread-locally for save verification. When
    the file has reached the size cap it is rotated to a timestamped
    archive first.
    """
    path = path or ANSWERS_FILE
    max_bytes = ANSWERS_MAX_BYTES if max_bytes is None else max_bytes
    record_id = uuid.uuid4().hex
    record = RECORD_TEMPLATE.format(
        record_id=record_id,
        query=query.strip(),
        assistant_answer=assistant_answer.strip(),
        web_search_answer=web_search_answer.strip(),
    )
    # The containing folder (data/) must exist before the lock file and the
    # answers file can be created inside it.
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path):
        _rotate_if_needed(path, max_bytes)
        with open(path, "a", encoding="utf-8") as f:
            f.write(record)
    _local.last_record_id = record_id
    logger.info("Appended support record %s to %s", record_id, path.name)
    return record_id


def reset_last_record_id() -> None:
    """Clear the thread-local marker before a new crew run."""
    _local.last_record_id = None


def get_last_record_id() -> str | None:
    """Record-ID written by this thread's most recent append, if any."""
    return getattr(_local, "last_record_id", None)


def record_exists(record_id: str | None, *, path: Path | None = None) -> bool:
    """True if a record with this ID is present in the answers file."""
    path = path or ANSWERS_FILE
    if not record_id or not path.exists():
        return False
    return f"Record-ID: {record_id}" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Custom file-writing tool (assigned ONLY to the Entry Agent)
# ---------------------------------------------------------------------------


@tool("Save Support Record")
def save_support_record(query: str, assistant_answer: str, web_search_answer: str) -> str:
    """Append a customer-support record to answers.txt using UTF-8 encoding.

    Pass the ORIGINAL user query, the COMPLETE unmodified Assistant answer,
    and the COMPLETE unmodified Web Search Assistant answer. The tool writes
    them in a fixed, readable format and never overwrites earlier records.
    """
    record_id = append_record(query, assistant_answer, web_search_answer)
    return (
        f"Record {record_id} successfully appended to "
        f"{ANSWERS_FILE.name} at {ANSWERS_FILE}"
    )


# ---------------------------------------------------------------------------
# Crew construction: 3 agents + 3 tasks wired sequentially
# ---------------------------------------------------------------------------


def build_crew() -> tuple[Crew, Task, Task, Task]:
    """Create the three agents, the three tasks, and the sequential crew.

    Returns the crew plus the individual task objects so the Streamlit UI
    can read each task's real output after kickoff. Agents are rebuilt per
    run on purpose: it keeps every kickoff stateless and thread-safe, at a
    cost that is negligible next to LLM latency.
    """
    # The web-search tool is created here and given ONLY to Agent 2.
    web_search_tool = SerperDevTool()

    # ----- AGENT 1: Assistant (direct answer, NO tools) --------------------
    assistant = Agent(
        role="Assistant",
        goal=(
            "Answer the customer's support query directly, clearly and "
            "helpfully using only your own knowledge."
        ),
        backstory=(
            "You are an experienced customer-support specialist. You answer "
            "questions from your own knowledge only. You never use web "
            "search and you never claim that you searched the web."
        ),
        tools=[],  # explicitly no tools
        llm=MODEL_NAME,
        max_execution_time=AGENT_MAX_EXECUTION_TIME,
        allow_delegation=False,
        verbose=False,
    )

    # ----- AGENT 2: Web Search Assistant (SerperDevTool only) --------------
    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Search the web for the customer's query and produce an answer "
            "grounded in the most relevant, current search results."
        ),
        backstory=(
            "You are a research-focused support specialist. For every query "
            "you first run a web search with your search tool, then write a "
            "clear standalone answer based on what the search returned. You "
            "never write files."
        ),
        tools=[web_search_tool],  # the ONLY agent with web search
        llm=MODEL_NAME,
        max_execution_time=AGENT_MAX_EXECUTION_TIME,
        allow_delegation=False,
        verbose=False,
    )

    # ----- AGENT 3: Entry Agent (file-writing tool only) -------------------
    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Record the original query and both earlier answers into "
            "answers.txt exactly as they were produced, then report both "
            "answers back in a structured form."
        ),
        backstory=(
            "You are a meticulous record keeper. You never invent, shorten "
            "or rewrite the answers you receive. You save them verbatim "
            "with your file-saving tool and return them unchanged. You "
            "never use web search."
        ),
        tools=[save_support_record],  # the ONLY agent with the file tool
        llm=MODEL_NAME,
        max_execution_time=AGENT_MAX_EXECUTION_TIME,
        allow_delegation=False,
        verbose=False,
    )

    # ----- TASK 1: Direct answer from the Assistant ------------------------
    direct_answer_task = Task(
        description=(
            "A customer submitted this support query:\n\n\"{query}\"\n\n"
            "Answer it directly using only your own knowledge. Do not use "
            "or mention web search. Provide a clear, complete, standalone "
            "answer the customer can act on."
        ),
        expected_output=(
            "Only your direct answer to the customer's query, as plain "
            "helpful text with plain ASCII punctuation (straight quotes). "
            "No preamble about how the answer was produced."
        ),
        agent=assistant,
    )

    # ----- TASK 2: Web-searched answer -------------------------------------
    web_search_task = Task(
        description=(
            "A customer submitted this support query:\n\n\"{query}\"\n\n"
            "Use your web-search tool to search for this query, review the "
            "results, and write a clear standalone answer grounded in the "
            "information you found. Do not copy or reference the previous "
            "agent's answer."
        ),
        expected_output=(
            "Only your answer to the customer's query based on the web "
            "search results, as plain helpful text with plain ASCII "
            "punctuation (straight quotes)."
        ),
        agent=web_search_assistant,
        # Explicit empty context: without this, CrewAI's sequential process
        # passes Task 1's answer here and the agent tends to copy it instead
        # of searching. Task 2 must answer from web search alone.
        context=[],
    )

    # ----- TASK 3: Save everything to answers.txt --------------------------
    # `context` passes the REAL outputs of Task 1 and Task 2 into this task,
    # which is the CrewAI-supported mechanism for sequential data flow.
    entry_task = Task(
        description=(
            "The original customer query was:\n\n\"{query}\"\n\n"
            "From the context you received: the FIRST context item is the "
            "Assistant's direct answer and the SECOND context item is the "
            "Web Search Assistant's answer.\n\n"
            "1. Call the 'Save Support Record' tool EXACTLY ONCE with three "
            "arguments: the original query, the complete Assistant answer, "
            "and the complete Web Search Assistant answer — all verbatim, "
            "with no truncation, summarising or rewriting.\n"
            "2. After the tool confirms the save, return the final "
            "structured result containing both answers unchanged, "
            "file_saved=true, and file_path='answers.txt'."
        ),
        expected_output=(
            "A structured object with fields assistant_answer, "
            "web_search_answer, file_saved and file_path, where both "
            "answers are preserved verbatim from the context."
        ),
        agent=entry_agent,
        context=[direct_answer_task, web_search_task],
        output_pydantic=SupportRecord,
    )

    crew = Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[direct_answer_task, web_search_task, entry_task],
        process=Process.sequential,  # required: strict 1 -> 2 -> 3 order
        verbose=False,
    )
    return crew, direct_answer_task, web_search_task, entry_task


def _usage_metric(result: object, name: str) -> int | None:
    """Read one token-usage field from a kickoff result, defensively."""
    usage = getattr(result, "token_usage", None)
    value = getattr(usage, name, None)
    return int(value) if isinstance(value, int | float) else None


def _attempt_run(query: str) -> RunResult:
    """One crew execution: build fresh agents/tasks, kick off, verify save.

    Runs entirely inside one worker thread, so the thread-local Record-ID
    written by the Entry Agent's tool is visible here. On failure, whether
    a record was already written is attached to the exception so the retry
    policy can veto a rerun that would duplicate it.
    """
    crew, task1, task2, _entry_task = build_crew()

    # Save verification, most reliable first: the Record-ID our thread's
    # tool call wrote; file-size delta kept as a fallback safety net.
    reset_last_record_id()
    size_before = ANSWERS_FILE.stat().st_size if ANSWERS_FILE.exists() else 0

    try:
        result = crew.kickoff(inputs={"query": query})
    except Exception as exc:
        setattr(exc, "record_written", get_last_record_id() is not None)  # noqa: B010
        raise

    assistant_answer = (task1.output.raw or "").strip() if task1.output else ""
    web_search_answer = (task2.output.raw or "").strip() if task2.output else ""

    record_id = get_last_record_id()
    if record_id is not None:
        file_saved = record_exists(record_id)
    else:
        size_after = ANSWERS_FILE.stat().st_size if ANSWERS_FILE.exists() else 0
        file_saved = size_after > size_before

    # Optional cross-check with the Entry Agent's structured output.
    entry_record = getattr(result, "pydantic", None)
    if entry_record is not None and not assistant_answer:
        assistant_answer = entry_record.assistant_answer.strip()
    if entry_record is not None and not web_search_answer:
        web_search_answer = entry_record.web_search_answer.strip()

    return RunResult(
        query=query,
        assistant_answer=assistant_answer,
        web_search_answer=web_search_answer,
        file_saved=file_saved,
        file_path=str(ANSWERS_FILE),
        total_tokens=_usage_metric(result, "total_tokens"),
        prompt_tokens=_usage_metric(result, "prompt_tokens"),
        completion_tokens=_usage_metric(result, "completion_tokens"),
    )


def run_support_crew(query: str) -> RunResult:
    """Run the sequential crew for one query and return the display data.

    The answers shown in the UI are taken from the REAL outputs of Task 1
    and Task 2 (task.output.raw), so the UI never depends on parsing the
    Entry Agent's prose. The Entry Agent still performs the file save and
    returns its own structured record as a cross-check.

    Each attempt runs in a worker thread with a hard overall deadline
    (RUN_TIMEOUT), so the UI can never spin forever even if a provider call
    hangs past the per-agent ceilings. Transient provider failures (rate
    limit, network, timeout) are retried with exponential backoff; anything
    else fails fast. A retry is vetoed if the failed attempt already wrote
    its record, so a record can never be duplicated. All failures surface
    as CrewRunError with a user-friendly, classified message.
    """
    # Log run metadata only — never the query text (it is user PII).
    logger.info("Crew run started (query length: %d chars)", len(query))
    started = time.monotonic()

    result = execute_with_retries(
        lambda: run_with_deadline(lambda: _attempt_run(query), timeout=RUN_TIMEOUT),
        attempts=MAX_ATTEMPTS,
        base_delay=RETRY_BASE_DELAY,
        abort_retry=lambda exc: bool(getattr(exc, "record_written", False)),
    )

    logger.info(
        "Crew run finished in %.1fs (file_saved=%s, tokens=%s prompt=%s completion=%s)",
        time.monotonic() - started,
        result.file_saved,
        result.total_tokens,
        result.prompt_tokens,
        result.completion_tokens,
    )
    return result


# ---------------------------------------------------------------------------
# UI styles: the base palette/fonts live in .streamlit/config.toml; this CSS
# adds the pieces native theming cannot express (gradient sidebar, hero,
# step cards, badges, result cards). Widget targeting uses stable `key=`
# classes. __ACTIVE_NAV__ is substituted with the active nav button's class.
# ---------------------------------------------------------------------------

_CSS_TEMPLATE = """
<style>
/* ---- app shell ---- */
/* normal arrow cursor everywhere so text never looks editable; only
   real inputs and clickable elements get their own cursors back */
.stApp, .stApp * { cursor: default !important; }
.stApp textarea, .stApp input { cursor: text !important; }
.stApp button, .stApp button *, .stApp a, .stApp a *,
.stApp [role="button"], .stApp [role="button"] *,
.stApp summary, .stApp summary * { cursor: pointer !important; }
/* hide the blinking text caret outside real inputs (e.g. when the
   browser's caret-browsing mode is on) */
.stApp { caret-color: transparent; }
.stApp textarea, .stApp input { caret-color: auto; }
/* decorative chrome is not selectable; answers stay copyable */
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
.sc-topbar, .sc-hero, .sc-steps, .sc-badges,
.sc-ask-label, .sc-success, .sc-saved-row, .sc-card-head, .sc-card-foot {
    user-select: none !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 88% 6%, rgba(139, 92, 246, .10), transparent 40%),
        radial-gradient(circle at 10% 96%, rgba(59, 130, 246, .07), transparent 38%),
        linear-gradient(160deg, #f8f6fe 0%, #f3f0fc 100%);
}
[data-testid="stHeader"] { background: transparent; }
/* hide Streamlit chrome, but NOT stToolbar itself — it holds the
   expand-sidebar chevron shown when the sidebar is collapsed */
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #221c4f 0%, #2d2569 45%, #4b3ecf 100%);
}
/* collapse / expand chevrons: always visible (not only on hover) */
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] button { color: rgba(255, 255, 255, .85) !important; }
[data-testid="stSidebarCollapseButton"] button:hover { color: #ffffff !important; }
[data-testid="stExpandSidebarButton"] { color: #4b3ecf !important; }
[data-testid="stSidebarUserContent"] { padding: 1.4rem 1rem 1.4rem; }

.sc-logo { text-align: center; padding: .4rem 0 1.6rem; }
.sc-logo-icon { width: 78px; margin: 0 auto .7rem; }
.sc-logo-icon img {
    width: 78px; height: 78px; display: block;
    filter: drop-shadow(0 0 16px rgba(167, 139, 250, .65));
}
.sc-logo-title { color: #ffffff; font-size: 1.35rem; font-weight: 800; letter-spacing: .01em; }
.sc-logo-sub { color: rgba(226, 220, 255, .75); font-size: .85rem; margin-top: .25rem; }

[data-testid="stSidebar"] .stButton button {
    width: 100%; justify-content: flex-start; gap: .3rem;
    background: transparent; border: none;
    color: rgba(255, 255, 255, .82); font-weight: 500;
    padding: .6rem .95rem; border-radius: 12px;
}
/* the label lives in a full-width inner wrapper that centers its content;
   left-align it so nav items match the mock */
[data-testid="stSidebar"] .stButton button > div {
    justify-content: flex-start; text-align: left;
}
[data-testid="stSidebar"] .stButton button:hover,
[data-testid="stSidebar"] .stButton button:focus:not(:active) {
    background: rgba(255, 255, 255, .10); color: #ffffff;
}
__ACTIVE_NAV__ button {
    background: rgba(255, 255, 255, .16) !important; color: #ffffff !important;
}

/* pin the privacy card to the bottom of the sidebar */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
    min-height: calc(100vh - 5rem);
    gap: .4rem;
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:last-child {
    margin-top: auto;
}

.sc-privacy {
    padding: 1rem 1.1rem; border-radius: 14px;
    background: rgba(255, 255, 255, .10);
    border: 1px solid rgba(255, 255, 255, .12);
}
.sc-privacy-title { color: #ffffff; font-weight: 700; font-size: .92rem; margin-bottom: .45rem; }
.sc-privacy-text { color: rgba(226, 220, 255, .80); font-size: .82rem; line-height: 1.45; }

/* ---- top bar / hero ---- */
.sc-topbar { display: flex; justify-content: flex-end; margin-bottom: .2rem; }
.sc-online {
    display: inline-flex; align-items: center; gap: .5rem;
    background: #ffffff; border-radius: 999px; padding: .45rem 1.1rem;
    font-size: .85rem; font-weight: 600; color: #333a56;
    box-shadow: 0 3px 12px rgba(60, 50, 150, .08);
}
.sc-dot {
    width: 9px; height: 9px; border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 6px rgba(34, 197, 94, .8);
}
.sc-hero { text-align: center; margin-bottom: 1.6rem; }
.sc-hero h1 {
    font-size: 2.35rem; font-weight: 800; color: #1b2050;
    margin: 0 0 .5rem; padding: 0; letter-spacing: -.01em;
}
.sc-hero p { color: #6b7280; font-size: 1.05rem; margin: 0; }
.sc-hero p b { color: #7c3aed; }

/* ---- agent step cards ---- */
.sc-steps {
    display: flex; align-items: center; gap: .6rem;
    margin-bottom: 1.6rem; flex-wrap: wrap;
}
.sc-step {
    flex: 1 1 240px; display: flex; align-items: center; gap: .7rem;
    background: #ffffff; border: 1px solid #eeeaf9; border-radius: 14px;
    padding: .85rem 1rem; box-shadow: 0 4px 14px rgba(60, 50, 150, .05);
}
.sc-step-num {
    width: 30px; height: 30px; flex: 0 0 30px; border-radius: 50%;
    color: #ffffff; font-weight: 700; font-size: .9rem;
    display: flex; align-items: center; justify-content: center;
}
.sc-step-icon {
    width: 40px; height: 40px; flex: 0 0 40px; border-radius: 12px;
    font-size: 20px; display: flex; align-items: center; justify-content: center;
}
.sc-step-title { font-weight: 700; color: #1f2544; font-size: .95rem; }
.sc-step-sub { color: #8a8fa3; font-size: .8rem; margin-top: .1rem; }
.sc-connector { flex: 0 1 34px; border-top: 2px dashed #d9d2f0; }

/* ---- query card ---- */
.st-key-query_card {
    background: #ffffff; border-radius: 18px; padding: 1.5rem 1.6rem;
    box-shadow: 0 10px 30px rgba(76, 63, 207, .08);
    border: 1px solid #f0edfb;
}
.sc-ask-label { font-weight: 600; color: #1f2544; font-size: 1rem; }
.st-key-query_card [data-baseweb="textarea"] {
    background: #fcfbff; border-color: #e6e1f7; border-radius: 12px;
}
.st-key-query_card textarea { background: #fcfbff; }

.st-key-query_card [data-testid="stForm"] { border: none; padding: 0; }
.st-key-query_card [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #8b5cf6 0%, #6d28d9 100%);
    border: none; color: #ffffff; font-weight: 600;
    padding: .55rem 1.5rem; border-radius: 10px;
    box-shadow: 0 6px 16px rgba(109, 40, 217, .35);
}
.st-key-query_card [data-testid="stFormSubmitButton"] button:hover {
    filter: brightness(1.07); color: #ffffff;
}

.sc-badges { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: .4rem; }
.sc-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: #f5f3fb; border: 1px solid #e9e4f7; border-radius: 9px;
    padding: .35rem .8rem; font-size: .78rem; color: #5b616e; font-weight: 500;
}
.sc-badge-green { background: #eaf7ef; border-color: #d4eedd; color: #1a7f37; }
.sc-badge code {
    background: transparent; color: inherit; padding: 0; font-size: .75rem;
}

/* ---- success banner ---- */
.sc-success {
    display: flex; align-items: center; gap: .9rem;
    background: #e6f8ee; border: 1px solid #d0f0dd; border-radius: 14px;
    padding: 1rem 1.3rem; margin: 1.4rem 0 1.5rem;
}
.sc-success-icon {
    width: 34px; height: 34px; flex: 0 0 34px; border-radius: 50%;
    background: #22c55e; color: #ffffff; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
}
.sc-success-title { color: #15803d; font-weight: 700; font-size: 1rem; }
.sc-success-sub { color: #3f7a55; font-size: .85rem; margin-top: .1rem; }
.sc-success-art { margin-left: auto; font-size: 1.7rem; }

/* ---- result cards ---- */
.st-key-assistant_card, .st-key-websearch_card {
    background: #ffffff; border: 1px solid #eeeaf9; border-radius: 16px;
    padding: 1.2rem 1.4rem; box-shadow: 0 8px 24px rgba(40, 40, 90, .06);
    height: 100%;
}
.st-key-assistant_card { border-left: 5px solid #a855f7; }
.st-key-websearch_card { border-left: 5px solid #3b82f6; }

.sc-card-head { display: flex; align-items: center; justify-content: space-between; gap: .6rem; }
.sc-card-title { font-size: 1.12rem; font-weight: 700; }
.sc-card-title.sc-purple { color: #9333ea; }
.sc-card-title.sc-blue { color: #3b82f6; }
.sc-chip {
    border-radius: 999px; padding: .28rem .85rem; font-size: .76rem;
    font-weight: 600; white-space: nowrap;
}
.sc-chip-purple { background: #f3e8ff; color: #7e22ce; }
.sc-chip-blue { background: #e0edff; color: #2563eb; }
.sc-card-foot {
    display: flex; justify-content: space-between; gap: .6rem;
    border-top: 1px solid #f0edf9; padding-top: .7rem; margin-top: .4rem;
    color: #8a8fa3; font-size: .8rem;
}

/* ---- saved-to-file banner ---- */
.st-key-saved_banner {
    background: #e8f1fd; border: 1px solid #d4e5fa; border-radius: 14px;
    padding: .9rem 1.2rem; margin-top: 1.4rem;
}
.sc-saved-row { display: flex; align-items: center; gap: .9rem; }
.sc-saved-icon {
    width: 30px; height: 30px; flex: 0 0 30px; border-radius: 50%;
    background: #3b82f6; color: #ffffff; font-weight: 800; font-size: .85rem;
    display: flex; align-items: center; justify-content: center;
}
.sc-saved-title { color: #2563eb; font-weight: 700; font-size: .95rem; }
.sc-saved-sub { color: #64748b; font-size: .83rem; margin-top: .1rem; }
.st-key-open_answers button {
    background: #ffffff; color: #2563eb; border: 1px solid #bcd6f7;
    font-weight: 600; border-radius: 10px; padding: .5rem 1.1rem;
}
.st-key-open_answers button:hover { border-color: #2563eb; color: #2563eb; }

/* ---- simple content card (history / about) ---- */
.st-key-content_card {
    background: #ffffff; border: 1px solid #eeeaf9; border-radius: 16px;
    padding: 1.4rem 1.6rem; box-shadow: 0 8px 24px rgba(40, 40, 90, .06);
}
</style>
"""

# Maps a view name to the sidebar nav button key whose pill is highlighted.
NAV_KEYS = {"new_query": "nav_new_query", "history": "nav_history", "about": "nav_about"}

# st.html strips <svg> elements during sanitization, so the logo is embedded
# as a base64 data-URI <img>, which passes through untouched.
_LOGO_SVG = """<svg viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="scBubble" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#8b7bff"/><stop offset="1" stop-color="#6d4df0"/>
</linearGradient>
<linearGradient id="scBand" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#a5b4fc"/><stop offset="1" stop-color="#93c5fd"/>
</linearGradient>
</defs>
<rect x="18" y="22" width="36" height="27" rx="11" fill="url(#scBubble)"/>
<path d="M28 47 l-2 10 11 -9 z" fill="url(#scBubble)"/>
<circle cx="29" cy="35.5" r="2.6" fill="#ffffff"/>
<circle cx="36" cy="35.5" r="2.6" fill="#ffffff"/>
<circle cx="43" cy="35.5" r="2.6" fill="#ffffff"/>
<path d="M14 38 v-7 c0 -12.15 9.85 -22 22 -22 c12.15 0 22 9.85 22 22 v7"
 fill="none" stroke="url(#scBand)" stroke-width="5.5" stroke-linecap="round"/>
<rect x="9" y="36" width="10" height="17" rx="5" fill="url(#scBand)"/>
<rect x="53" y="36" width="10" height="17" rx="5" fill="url(#scBand)"/>
</svg>"""

_LOGO_B64 = base64.b64encode(_LOGO_SVG.encode("utf-8")).decode("ascii")

STEPS_HTML = """
<div class="sc-steps">
    <div class="sc-step">
        <div class="sc-step-num" style="background:#7c5cfc;">1</div>
        <div class="sc-step-icon" style="background:#f1ebff;">🧠</div>
        <div>
            <div class="sc-step-title">Assistant</div>
            <div class="sc-step-sub">Answers from knowledge</div>
        </div>
    </div>
    <div class="sc-connector"></div>
    <div class="sc-step">
        <div class="sc-step-num" style="background:#3b82f6;">2</div>
        <div class="sc-step-icon" style="background:#e6f0fe;">🔍</div>
        <div>
            <div class="sc-step-title">Web Search Assistant</div>
            <div class="sc-step-sub">Searches the web (Serper)</div>
        </div>
    </div>
    <div class="sc-connector"></div>
    <div class="sc-step">
        <div class="sc-step-num" style="background:#22c55e;">3</div>
        <div class="sc-step-icon" style="background:#e6f8ee;">📄</div>
        <div>
            <div class="sc-step-title">Entry Agent</div>
            <div class="sc-step-sub">Saves results to answers.txt</div>
        </div>
    </div>
</div>
"""


def inject_css(active_view: str) -> None:
    """Deliver the stylesheet with the active nav pill substituted in."""
    st.html(_CSS_TEMPLATE.replace("__ACTIVE_NAV__", f".st-key-{NAV_KEYS[active_view]}"))


def logo_html() -> str:
    return f"""
        <div class="sc-logo">
            <div class="sc-logo-icon">
                <img src="data:image/svg+xml;base64,{_LOGO_B64}"
                     alt="Support Crew logo" />
            </div>
            <div class="sc-logo-title">Support Crew</div>
            <div class="sc-logo-sub">AI-Powered Help</div>
        </div>
    """


def completed_label(timestamp: float) -> str:
    """Human-friendly 'Completed …' label for a result timestamp."""
    minutes = int((time.time() - timestamp) // 60)
    if minutes < 1:
        return "Completed just now"
    if minutes == 1:
        return "Completed 1 min ago"
    return f"Completed {minutes} min ago"


def render_result_cards(res: RunResult, timestamp: float) -> None:
    """The two side-by-side answer cards (Assistant + Web Search)."""
    completed = completed_label(timestamp)
    cards = (
        (
            "assistant_card",
            '<div class="sc-card-title sc-purple">🧠 Assistant Answer</div>'
            '<span class="sc-chip sc-chip-purple">Direct Answer</span>',
            res.assistant_answer,
            "Assistant",
        ),
        (
            "websearch_card",
            '<div class="sc-card-title sc-blue">🌐 Web Search Answer</div>'
            '<span class="sc-chip sc-chip-blue">Web Results</span>',
            res.web_search_answer,
            "Web Search Assistant",
        ),
    )
    for col, (key, head, answer, agent_name) in zip(
        st.columns(2, gap="medium"), cards, strict=True
    ):
        with col, st.container(key=key):
            st.html(f'<div class="sc-card-head">{head}</div>')
            st.markdown(answer or "_No answer was produced._")
            st.html(
                f"""
                <div class="sc-card-foot">
                    <span>🤖 Agent: {agent_name}</span>
                    <span>🕐 {completed}</span>
                </div>
                """
            )


# ---------------------------------------------------------------------------
# UI views: session state, sidebar navigation, and the three pages
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Session state keeps completed results visible across Streamlit reruns."""
    if "support_result" not in st.session_state:
        st.session_state.support_result = None
    if "support_error" not in st.session_state:
        st.session_state.support_error = None
    if "view" not in st.session_state:
        st.session_state.view = "new_query"
    if "history" not in st.session_state:
        st.session_state.history = []


def _set_view(view: str) -> None:
    st.session_state.view = view
    # "New Query" starts a fresh session: previous output, errors and the
    # typed query are cleared (past runs remain available under History).
    if view == "new_query":
        st.session_state.support_result = None
        st.session_state.support_error = None
        st.session_state.clear_query = True


def render_sidebar() -> None:
    with st.sidebar:
        st.html(logo_html())
        st.button(
            "New Query", icon=":material/chat:", key="nav_new_query",
            width="stretch", on_click=_set_view, args=("new_query",),
        )
        st.button(
            "History", icon=":material/description:", key="nav_history",
            width="stretch", on_click=_set_view, args=("history",),
        )
        st.button(
            "About", icon=":material/info:", key="nav_about",
            width="stretch", on_click=_set_view, args=("about",),
        )
        st.html(
            """
            <div class="sc-privacy">
                <div class="sc-privacy-title">🛡️ Your data is safe</div>
                <div class="sc-privacy-text">We value your privacy. Queries are
                processed securely.</div>
            </div>
            """
        )


def render_header() -> None:
    st.html(
        """
        <div class="sc-topbar">
            <span class="sc-online"><span class="sc-dot"></span>System Online</span>
        </div>
        <div class="sc-hero">
            <h1>🤝 Multi-Agent Customer Support System</h1>
            <p>Three AI agents work together to give you the
            <b>best support experience.</b></p>
        </div>
        """
    )


def render_query_card() -> None:
    """The input card; handles the Run submit and reruns once results exist."""
    # Blank the box for the next query after a successful run. Widget state
    # can only be changed before the widget is instantiated, hence the flag.
    if st.session_state.pop("clear_query", False):
        st.session_state.query_input = ""

    with st.container(key="query_card"):
        st.html('<div class="sc-ask-label">Ask your question or describe your issue</div>')
        # A form makes Ctrl+Enter in the text area submit the query.
        with st.form("query_form", border=False):
            query = st.text_area(
                "Ask your question or describe your issue",
                placeholder="e.g. How do I reset my password?",
                height=110,
                label_visibility="collapsed",
                key="query_input",
            )
            with st.container(horizontal=True, horizontal_alignment="right"):
                run_clicked = st.form_submit_button(
                    "Run Support Crew", icon="🚀", type="primary"
                )
        st.html(
            """
            <div class="sc-badges">
                <span class="sc-badge">🔒 Secure &amp; Private</span>
                <span class="sc-badge">⚡ Powered by CrewAI</span>
                <span class="sc-badge">🔍 Web Search with Serper</span>
                <span class="sc-badge sc-badge-green">📝 Results saved to
                <code>answers.txt</code></span>
            </div>
            """
        )

    if run_clicked:
        _handle_run(query)


def _handle_run(query: str) -> None:
    """Validate the submitted query and, if valid, execute the crew."""
    st.session_state.support_result = None
    st.session_state.support_error = None

    # --- Input validation: never invoke CrewAI on a bad query. ------------
    if not query or not query.strip():
        st.warning("Please enter a customer-support query before running the crew.")
        return
    if len(query) > MAX_QUERY_CHARS:
        st.warning(
            f"Your query is {len(query):,} characters long; the maximum is "
            f"{MAX_QUERY_CHARS:,}. Please shorten it and try again."
        )
        return

    # --- API-key validation (names only, values never shown). -------------
    missing = missing_api_keys()
    if missing:
        st.error(
            "Missing required environment variable(s): "
            f"**{', '.join(missing)}**. Please set them in your shell "
            "and restart the app. See README.md for setup commands."
        )
        return

    try:
        with st.spinner("🤖 Agents are working — this can take a minute…"):
            result = run_support_crew(query.strip())
        result.completed_at = time.time()
        st.session_state.support_result = result
        st.session_state.history.append(result)
        st.session_state.clear_query = True
    except CrewRunError as exc:
        # Classified failure (auth / rate limit / network / timeout /
        # unknown) with a user-safe message; retries already happened.
        logger.error("Crew run failed (%s)", exc.kind.value, exc_info=True)
        st.session_state.support_error = exc.user_message
    except Exception as exc:  # noqa: BLE001 - last-resort friendly message
        logger.exception("Crew run failed unexpectedly")
        st.session_state.support_error = (
            "Something went wrong while running the crew: "
            f"{type(exc).__name__}: {exc}"
        )
    # Rerun so the fresh output renders above the query card.
    st.rerun()


def render_outputs() -> None:
    """Results / errors, rendered from session state on every rerun."""
    if st.session_state.support_error:
        st.error(st.session_state.support_error)
        st.info(
            "Check that your API keys are valid, your internet connection is "
            "up, and then try again."
        )

    if st.session_state.support_result:
        res = st.session_state.support_result

        if res.file_saved:
            st.html(
                """
                <div class="sc-success">
                    <div class="sc-success-icon">✔</div>
                    <div>
                        <div class="sc-success-title">Crew finished successfully!</div>
                        <div class="sc-success-sub">All agents completed their tasks.</div>
                    </div>
                    <div class="sc-success-art">📄✅</div>
                </div>
                """
            )
        else:
            st.warning(
                "The crew finished, but no new record was detected in "
                "answers.txt. Please check the file manually."
            )

        render_result_cards(res, res.completed_at or time.time())

        if res.file_saved:
            with st.container(key="saved_banner"):
                left, right = st.columns([4, 1], vertical_alignment="center")
                with left:
                    st.html(
                        """
                        <div class="sc-saved-row">
                            <div class="sc-saved-icon">i</div>
                            <div>
                                <div class="sc-saved-title">Results saved to answers.txt</div>
                                <div class="sc-saved-sub">Your query and both answers
                                have been securely saved.</div>
                            </div>
                        </div>
                        """
                    )
                with right:
                    st.download_button(
                        "Open answers.txt",
                        icon=":material/folder_open:",
                        data=ANSWERS_FILE.read_bytes() if ANSWERS_FILE.exists() else b"",
                        file_name="answers.txt",
                        mime="text/plain",
                        key="open_answers",
                        width="stretch",
                    )


def render_new_query_view() -> None:
    st.html(STEPS_HTML)
    # Once an output (or error) exists, it takes the top spot and the query
    # card moves below it, ready for the next question.
    if st.session_state.support_result or st.session_state.support_error:
        render_outputs()
        render_query_card()
    else:
        render_query_card()


def render_history_view() -> None:
    with st.container(key="content_card"):
        st.markdown("### :material/description: Query history")
        if not st.session_state.history:
            st.info(
                "No queries yet in this session. Run one from **New Query** "
                "and it will show up here."
            )
        else:
            for i, item in enumerate(reversed(st.session_state.history), start=1):
                with st.expander(f"💬 {item.query}", expanded=(i == 1)):
                    st.markdown("**🧠 Assistant answer**")
                    st.markdown(item.assistant_answer or "_No answer was produced._")
                    st.markdown("**🌐 Web search answer**")
                    st.markdown(item.web_search_answer or "_No answer was produced._")


def render_about_view() -> None:
    st.html(STEPS_HTML)
    with st.container(key="content_card"):
        st.markdown("### :material/info: About this app")
        st.markdown(
            "Three CrewAI agents handle your query **sequentially**:\n\n"
            "1. **Assistant** — answers directly from its own knowledge.\n"
            "2. **Web Search Assistant** — searches the web (Serper) and "
            "answers from the results.\n"
            "3. **Entry Agent** — saves the query and both answers to "
            "`answers.txt`.\n\n"
            "Built with [CrewAI](https://www.crewai.com/) and "
            "[Streamlit](https://streamlit.io/)."
        )


def render_current_view() -> None:
    if st.session_state.view == "history":
        render_history_view()
    elif st.session_state.view == "about":
        render_about_view()
    else:
        render_new_query_view()


# ---------------------------------------------------------------------------
# Entry point: `streamlit run app.py` executes this with __name__ set to
# "__main__" (as does the AppTest harness); a plain `import app` (e.g. from
# the unit tests) does not render any UI.
# ---------------------------------------------------------------------------


def main() -> None:
    # st.set_page_config must be the first Streamlit call of the script run.
    st.set_page_config(
        page_title="Multi-Agent Customer Support System",
        page_icon="🤝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Idempotent across Streamlit reruns: basicConfig is a no-op once the
    # root logger has handlers. API-key VALUES are never logged anywhere.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    init_session_state()
    inject_css(st.session_state.view)
    render_sidebar()
    render_header()
    render_current_view()


if __name__ == "__main__":
    main()
