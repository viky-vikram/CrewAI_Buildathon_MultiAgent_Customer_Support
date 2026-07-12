"""
Multi-Agent Customer Support System
===================================
Streamlit entry point. All application logic lives in the `support_crew`
package:

    support_crew.config   -- settings, env loading, validation
    support_crew.crew     -- the three sequential CrewAI agents + tasks
    support_crew.storage  -- locked answers.txt persistence with record IDs
    support_crew.ui       -- views, components, and the stylesheet

Run with:
    streamlit run app.py
"""

import logging

import streamlit as st

from support_crew.ui import components, views

# st.set_page_config must be the first Streamlit call of the script run.
st.set_page_config(
    page_title="Multi-Agent Customer Support System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Idempotent across Streamlit reruns: basicConfig is a no-op once the root
# logger has handlers. API-key VALUES are never logged anywhere.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

views.init_session_state()
components.inject_css(st.session_state.view)
views.render_sidebar()
views.render_header()
views.render_current_view()
