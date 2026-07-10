"""
Multi-Agent Customer Support System
===================================
A CrewAI application with exactly three agents running sequentially:

    1. Assistant            -> answers the query from its own LLM knowledge
    2. Web Search Assistant -> answers the same query using Serper web search
    3. Entry Agent          -> saves the query + both answers to answers.txt

The crew is coordinated with `process=Process.sequential`, and Task 3
receives the outputs of Task 1 and Task 2 through CrewAI task `context`.

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# answers.txt lives next to app.py and is created automatically on first save.
ANSWERS_FILE = Path(__file__).resolve().parent / "answers.txt"

RECORD_TEMPLATE = (
    "============================================================\n"
    "MULTI-AGENT CUSTOMER SUPPORT RESPONSE\n"
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


# ---------------------------------------------------------------------------
# Structured output model for the Entry Agent's final task
# ---------------------------------------------------------------------------

class SupportRecord(BaseModel):
    """Reliably parseable final output returned by the Entry Agent."""

    assistant_answer: str
    web_search_answer: str
    file_saved: bool
    file_path: str


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
    record = RECORD_TEMPLATE.format(
        query=query.strip(),
        assistant_answer=assistant_answer.strip(),
        web_search_answer=web_search_answer.strip(),
    )
    # Append mode ("a") preserves earlier support records; UTF-8 is explicit.
    with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(record)
    return f"Record successfully appended to {ANSWERS_FILE.name} at {ANSWERS_FILE}"


# ---------------------------------------------------------------------------
# Environment-variable validation (values are never printed or displayed)
# ---------------------------------------------------------------------------

def missing_api_keys() -> list[str]:
    """Return the names of any required API-key environment variables not set."""
    required = ("OPENAI_API_KEY", "SERPER_API_KEY")
    return [name for name in required if not os.environ.get(name, "").strip()]


# ---------------------------------------------------------------------------
# Crew construction: 3 agents + 3 tasks wired sequentially
# ---------------------------------------------------------------------------

def build_crew() -> tuple[Crew, Task, Task, Task]:
    """Create the three agents, the three tasks, and the sequential crew.

    Returns the crew plus the individual task objects so the Streamlit UI
    can read each task's real output after kickoff.
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


# ---------------------------------------------------------------------------
# Crew execution helper
# ---------------------------------------------------------------------------

def run_support_crew(query: str) -> dict:
    """Run the sequential crew for one query and return the display data.

    The answers shown in the UI are taken from the REAL outputs of Task 1
    and Task 2 (task.output.raw), so the UI never depends on parsing the
    Entry Agent's prose. The Entry Agent still performs the file save and
    returns its own structured record as a cross-check.
    """
    crew, task1, task2, task3 = build_crew()

    # File size before the run lets us verify the save actually happened.
    size_before = ANSWERS_FILE.stat().st_size if ANSWERS_FILE.exists() else 0

    result = crew.kickoff(inputs={"query": query})

    assistant_answer = (task1.output.raw or "").strip() if task1.output else ""
    web_search_answer = (task2.output.raw or "").strip() if task2.output else ""

    size_after = ANSWERS_FILE.stat().st_size if ANSWERS_FILE.exists() else 0
    file_saved = size_after > size_before

    # Optional cross-check with the Entry Agent's structured output.
    entry_record = getattr(result, "pydantic", None)
    if entry_record is not None and not assistant_answer:
        assistant_answer = entry_record.assistant_answer.strip()
    if entry_record is not None and not web_search_answer:
        web_search_answer = entry_record.web_search_answer.strip()

    return {
        "query": query,
        "assistant_answer": assistant_answer,
        "web_search_answer": web_search_answer,
        "file_saved": file_saved,
        "file_path": str(ANSWERS_FILE),
    }


# ---------------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Customer Support System",
    page_icon="🤝",
    layout="centered",
)

st.title("🤝 Multi-Agent Customer Support System")
st.markdown(
    "Three CrewAI agents handle your query **sequentially**:\n\n"
    "1. **Assistant** — answers directly from its own knowledge.\n"
    "2. **Web Search Assistant** — searches the web (Serper) and answers "
    "from the results.\n"
    "3. **Entry Agent** — saves the query and both answers to `answers.txt`."
)

# Session state keeps completed results visible across Streamlit reruns.
if "support_result" not in st.session_state:
    st.session_state.support_result = None
if "support_error" not in st.session_state:
    st.session_state.support_error = None

query = st.text_area(
    "Customer-support query or task",
    placeholder="e.g. How do I reset my password?",
    height=100,
)

if st.button("Run Support Crew", type="primary"):
    st.session_state.support_result = None
    st.session_state.support_error = None

    # --- Input validation: never invoke CrewAI on a blank query. ----------
    if not query or not query.strip():
        st.warning("Please enter a customer-support query before running the crew.")
    else:
        # --- API-key validation (names only, values never shown). ---------
        missing = missing_api_keys()
        if missing:
            st.error(
                "Missing required environment variable(s): "
                f"**{', '.join(missing)}**. Please set them in your shell "
                "and restart the app. See README.md for setup commands."
            )
        else:
            status = st.status("Running the support crew…", expanded=True)
            try:
                status.write("1) Assistant is preparing a direct answer…")
                status.write("2) Web Search Assistant is researching the query…")
                status.write("3) Entry Agent is saving the result…")
                with st.spinner("Agents are working — this can take a minute…"):
                    st.session_state.support_result = run_support_crew(query.strip())
                status.update(label="Crew finished.", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 - show a friendly message
                status.update(label="Crew failed.", state="error", expanded=False)
                st.session_state.support_error = (
                    "Something went wrong while running the crew: "
                    f"{type(exc).__name__}: {exc}"
                )

# --- Results / errors (rendered from session state on every rerun) --------
if st.session_state.support_error:
    st.error(st.session_state.support_error)
    st.info(
        "Check that your API keys are valid, your internet connection is "
        "up, and then try again."
    )

if st.session_state.support_result:
    res = st.session_state.support_result

    st.divider()
    st.subheader("🧠 Assistant Answer")
    st.markdown(res["assistant_answer"] or "_No answer was produced._")

    st.subheader("🌐 Web Search Answer")
    st.markdown(res["web_search_answer"] or "_No answer was produced._")

    st.divider()
    if res["file_saved"]:
        st.success("✅ The query and both answers were saved to **answers.txt**.")
    else:
        st.warning(
            "The crew finished, but no new record was detected in "
            "answers.txt. Please check the file manually."
        )
