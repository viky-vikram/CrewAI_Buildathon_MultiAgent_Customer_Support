"""Crew construction and execution: 3 agents + 3 tasks wired sequentially.

    1. Assistant            -> answers the query from its own LLM knowledge
    2. Web Search Assistant -> answers the same query using Serper web search
    3. Entry Agent          -> saves the query + both answers to answers.txt

The crew is coordinated with `process=Process.sequential`, and Task 3
receives the outputs of Task 1 and Task 2 through CrewAI task `context`.
"""

import logging
import time

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

from . import config, errors, storage
from .models import RunResult, SupportRecord
from .tools import save_support_record

logger = logging.getLogger(__name__)


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
        llm=config.MODEL_NAME,
        max_execution_time=config.AGENT_MAX_EXECUTION_TIME,
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
        llm=config.MODEL_NAME,
        max_execution_time=config.AGENT_MAX_EXECUTION_TIME,
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
        llm=config.MODEL_NAME,
        max_execution_time=config.AGENT_MAX_EXECUTION_TIME,
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
    storage.reset_last_record_id()
    size_before = config.ANSWERS_FILE.stat().st_size if config.ANSWERS_FILE.exists() else 0

    try:
        result = crew.kickoff(inputs={"query": query})
    except Exception as exc:
        setattr(exc, "record_written", storage.get_last_record_id() is not None)  # noqa: B010
        raise

    assistant_answer = (task1.output.raw or "").strip() if task1.output else ""
    web_search_answer = (task2.output.raw or "").strip() if task2.output else ""

    record_id = storage.get_last_record_id()
    if record_id is not None:
        file_saved = storage.record_exists(record_id)
    else:
        size_after = config.ANSWERS_FILE.stat().st_size if config.ANSWERS_FILE.exists() else 0
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
        file_path=str(config.ANSWERS_FILE),
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

    result = errors.execute_with_retries(
        lambda: errors.run_with_deadline(
            lambda: _attempt_run(query), timeout=config.RUN_TIMEOUT
        ),
        attempts=config.MAX_ATTEMPTS,
        base_delay=config.RETRY_BASE_DELAY,
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
