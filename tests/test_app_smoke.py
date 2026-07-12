"""End-to-end view smoke tests via Streamlit's AppTest (no browser, no LLM).

These execute the real app script headlessly: session-state wiring, sidebar
navigation, view switching, and input validation — everything except an
actual crew run (which needs the network).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
    keys = {btn.key for btn in app.sidebar.button}
    assert keys == {"nav_new_query", "nav_history", "nav_about"}
    assert len(app.text_area) == 1
    assert app.text_area[0].key == "query_input"


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
