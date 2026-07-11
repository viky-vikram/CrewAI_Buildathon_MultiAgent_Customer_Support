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

import base64
import os
import time
from pathlib import Path

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from pydantic import BaseModel

# Load API keys from a local .env file (NAME=value lines) into environment
# variables, so they don't have to be set in the shell for every session.
# Shell-set variables still work and take precedence over the .env file.
load_dotenv()

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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state keeps completed results visible across Streamlit reruns.
if "support_result" not in st.session_state:
    st.session_state.support_result = None
if "support_error" not in st.session_state:
    st.session_state.support_error = None
if "view" not in st.session_state:
    st.session_state.view = "new_query"
if "history" not in st.session_state:
    st.session_state.history = []

# --- Custom look & feel ----------------------------------------------------
# The base palette/fonts live in .streamlit/config.toml; this CSS adds the
# pieces native theming cannot express (gradient sidebar, hero, step cards,
# badges, result cards). Widget targeting uses stable `key=` classes.

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
.sc-steps { display: flex; align-items: center; gap: .6rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
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

_NAV_KEYS = {"new_query": "nav_new_query", "history": "nav_history", "about": "nav_about"}

st.html(
    _CSS_TEMPLATE.replace(
        "__ACTIVE_NAV__", f".st-key-{_NAV_KEYS[st.session_state.view]}"
    )
)


def _set_view(view: str) -> None:
    st.session_state.view = view


def _completed_label(timestamp: float) -> str:
    """Human-friendly 'Completed …' label for a result timestamp."""
    minutes = int((time.time() - timestamp) // 60)
    if minutes < 1:
        return "Completed just now"
    if minutes == 1:
        return "Completed 1 min ago"
    return f"Completed {minutes} min ago"


# --- Sidebar ---------------------------------------------------------------

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

with st.sidebar:
    st.html(
        f"""
        <div class="sc-logo">
            <div class="sc-logo-icon">
                <img src="data:image/svg+xml;base64,{_LOGO_B64}"
                     alt="Support Crew logo" />
            </div>
            <div class="sc-logo-title">Support Crew</div>
            <div class="sc-logo-sub">AI-Powered Help</div>
        </div>
        """
    )
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

# --- Header (all views) ----------------------------------------------------

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

_STEPS_HTML = """
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


def render_result_cards(res: dict, timestamp: float) -> None:
    """The two side-by-side answer cards (Assistant + Web Search)."""
    completed = _completed_label(timestamp)
    col1, col2 = st.columns(2, gap="medium")

    with col1, st.container(key="assistant_card"):
        st.html(
            """
            <div class="sc-card-head">
                <div class="sc-card-title sc-purple">🧠 Assistant Answer</div>
                <span class="sc-chip sc-chip-purple">Direct Answer</span>
            </div>
            """
        )
        st.markdown(res["assistant_answer"] or "_No answer was produced._")
        st.html(
            f"""
            <div class="sc-card-foot">
                <span>🤖 Agent: Assistant</span>
                <span>🕐 {completed}</span>
            </div>
            """
        )

    with col2, st.container(key="websearch_card"):
        st.html(
            """
            <div class="sc-card-head">
                <div class="sc-card-title sc-blue">🌐 Web Search Answer</div>
                <span class="sc-chip sc-chip-blue">Web Results</span>
            </div>
            """
        )
        st.markdown(res["web_search_answer"] or "_No answer was produced._")
        st.html(
            f"""
            <div class="sc-card-foot">
                <span>🤖 Agent: Web Search Assistant</span>
                <span>🕐 {completed}</span>
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
        st.session_state.support_result = None
        st.session_state.support_error = None

        # --- Input validation: never invoke CrewAI on a blank query. ------
        if not query or not query.strip():
            st.warning("Please enter a customer-support query before running the crew.")
        else:
            # --- API-key validation (names only, values never shown). -----
            missing = missing_api_keys()
            if missing:
                st.error(
                    "Missing required environment variable(s): "
                    f"**{', '.join(missing)}**. Please set them in your shell "
                    "and restart the app. See README.md for setup commands."
                )
            else:
                try:
                    with st.spinner("🤖 Agents are working — this can take a minute…"):
                        result = run_support_crew(query.strip())
                    result["completed_at"] = time.time()
                    st.session_state.support_result = result
                    st.session_state.history.append(result)
                    st.session_state.clear_query = True
                except Exception as exc:  # noqa: BLE001 - show a friendly message
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

        if res["file_saved"]:
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

        render_result_cards(res, res.get("completed_at", time.time()))

        if res["file_saved"]:
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
    st.html(_STEPS_HTML)
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
                with st.expander(f"💬 {item['query']}", expanded=(i == 1)):
                    st.markdown("**🧠 Assistant answer**")
                    st.markdown(item["assistant_answer"] or "_No answer was produced._")
                    st.markdown("**🌐 Web search answer**")
                    st.markdown(item["web_search_answer"] or "_No answer was produced._")


def render_about_view() -> None:
    st.html(_STEPS_HTML)
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


if st.session_state.view == "history":
    render_history_view()
elif st.session_state.view == "about":
    render_about_view()
else:
    render_new_query_view()
