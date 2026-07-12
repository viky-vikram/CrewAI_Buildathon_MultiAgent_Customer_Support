"""Reusable presentation pieces: stylesheet, logo, step cards, result cards.

The base palette/fonts live in .streamlit/config.toml; styles.css adds the
pieces native theming cannot express (gradient sidebar, hero, step cards,
badges, result cards). Widget targeting uses stable `key=` classes.
"""

import base64
import functools
import time
from pathlib import Path

import streamlit as st

from .. import config
from ..models import RunResult

_UI_DIR = Path(__file__).resolve().parent

# Maps a view name to the sidebar nav button key whose pill is highlighted.
NAV_KEYS = {"new_query": "nav_new_query", "history": "nav_history", "about": "nav_about"}

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


@functools.lru_cache(maxsize=1)
def _css_template() -> str:
    return (_UI_DIR / "styles.css").read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _logo_b64() -> str:
    # st.html strips <svg> elements during sanitization, so the logo is
    # embedded as a base64 data-URI <img>, which passes through untouched.
    svg = (config.PROJECT_ROOT / "static" / "logo.svg").read_bytes()
    return base64.b64encode(svg).decode("ascii")


def inject_css(active_view: str) -> None:
    """Deliver the stylesheet with the active nav pill substituted in."""
    css = _css_template().replace(
        "__ACTIVE_NAV__", f".st-key-{NAV_KEYS[active_view]}"
    )
    st.html(f"<style>\n{css}\n</style>")


def logo_html() -> str:
    return f"""
        <div class="sc-logo">
            <div class="sc-logo-icon">
                <img src="data:image/svg+xml;base64,{_logo_b64()}"
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
