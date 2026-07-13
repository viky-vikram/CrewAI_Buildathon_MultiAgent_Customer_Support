"""End-to-end view smoke tests via Streamlit's AppTest (no browser, no LLM).

These execute the real app script headlessly: session-state wiring, sidebar
navigation, view switching, and input validation — everything except an
actual crew run (which needs the network).
"""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import RunResult

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    return at.run()


def test_app_boots_without_exception(app):
    assert not app.exception
    assert app.session_state["view"] == "new_query"
    assert app.session_state["support_result"] is None
    assert app.session_state["history"] == []


def test_sidebar_has_three_nav_buttons_and_query_form(app):
    # No recents before any run — only the three nav buttons.
    keys = {btn.key for btn in app.sidebar.button}
    assert keys == {"nav_new_query", "nav_history", "nav_about"}
    assert len(app.text_area) == 1
    assert app.text_area[0].key == "query_input"


def _past_session(query: str) -> RunResult:
    return RunResult(
        query=query,
        assistant_answer="Direct answer.",
        web_search_answer="Web answer.",
        file_saved=True,
        file_path="data/answers.txt",
        completed_at=time.time(),
    )


def test_past_sessions_appear_under_recents_and_reopen(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.session_state["history"] = [_past_session("How do I log in?")]
    at.run()
    assert not at.exception

    # The past session is listed in the sidebar Recents…
    recents = [b for b in at.sidebar.button if (b.key or "").startswith("recent_")]
    assert [b.label for b in recents] == ["How do I log in?"]

    # …and clicking it restores that session's answers.
    recents[0].click().run()
    assert not at.exception
    assert at.session_state["view"] == "new_query"
    assert at.session_state["support_result"].query == "How do I log in?"


def test_recents_show_newest_first_and_cap_at_five(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.session_state["history"] = [_past_session(f"Query {i}") for i in range(7)]
    at.run()
    assert not at.exception

    recents = [b for b in at.sidebar.button if (b.key or "").startswith("recent_")]
    assert [b.label for b in recents] == [
        "Query 6", "Query 5", "Query 4", "Query 3", "Query 2"
    ]


def test_nav_switches_to_history_and_back(app):
    app.sidebar.button(key="nav_history").click().run()
    assert not app.exception
    assert app.session_state["view"] == "history"

    app.sidebar.button(key="nav_new_query").click().run()
    assert not app.exception
    assert app.session_state["view"] == "new_query"


def test_about_view_renders(app):
    app.sidebar.button(key="nav_about").click().run()
    assert not app.exception
    assert app.session_state["view"] == "about"


def test_blank_query_warns_and_never_runs_the_crew(app):
    submits = [b for b in app.button if b.key and "FormSubmitter" in b.key]
    assert len(submits) == 1
    submits[0].click().run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "enter a customer-support query" in app.warning[0].value
    assert app.session_state["support_result"] is None


def test_overlong_query_is_rejected(app, monkeypatch):
    app.text_area(key="query_input").set_value("x" * 3000).run()
    submits = [b for b in app.button if b.key and "FormSubmitter" in b.key]
    submits[0].click().run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "maximum" in app.warning[0].value
    assert app.session_state["support_result"] is None
