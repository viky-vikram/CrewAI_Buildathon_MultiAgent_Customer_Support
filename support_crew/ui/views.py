"""Streamlit views: session state, sidebar navigation, and the three pages."""

import logging
import time

import streamlit as st

from .. import config
from ..crew import run_support_crew
from ..errors import CrewRunError
from . import components

logger = logging.getLogger(__name__)


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
        st.html(components.logo_html())
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
    if len(query) > config.MAX_QUERY_CHARS:
        st.warning(
            f"Your query is {len(query):,} characters long; the maximum is "
            f"{config.MAX_QUERY_CHARS:,}. Please shorten it and try again."
        )
        return

    # --- API-key validation (names only, values never shown). -------------
    missing = config.missing_api_keys()
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

        components.render_result_cards(res, res.completed_at or time.time())

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
                        data=config.ANSWERS_FILE.read_bytes()
                        if config.ANSWERS_FILE.exists()
                        else b"",
                        file_name="answers.txt",
                        mime="text/plain",
                        key="open_answers",
                        width="stretch",
                    )


def render_new_query_view() -> None:
    st.html(components.STEPS_HTML)
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
    st.html(components.STEPS_HTML)
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
