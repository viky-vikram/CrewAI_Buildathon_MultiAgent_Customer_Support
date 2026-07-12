"""Central configuration: every environmental assumption lives here.

Settings resolve in precedence order:
    1. shell environment variables,
    2. a local .env file (NAME=value lines, loaded by python-dotenv),
    3. Streamlit secrets (.streamlit/secrets.toml locally, or the Secrets
       panel on Streamlit Community Cloud) — filled into the environment
       only for names not already set.

Key VALUES are never printed, logged, or displayed anywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

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
    values always win. Silently a no-op when no secrets file exists or
    Streamlit is unavailable.
    """
    try:
        import streamlit as st

        for name in _ENV_NAMES:
            if name not in os.environ and name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except Exception:  # noqa: BLE001 - missing secrets.toml is normal
        pass


_fill_env_from_streamlit_secrets()

# Project root is the folder containing app.py (one level above this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# answers.txt lives next to app.py and is created automatically on first save.
ANSWERS_FILE = PROJECT_ROOT / "answers.txt"

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
