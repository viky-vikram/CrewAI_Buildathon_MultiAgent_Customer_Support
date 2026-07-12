# Multi-Agent Customer Support System

A **CrewAI** application built for the Gen AI Architect Program **Weekly Buildathon**.

## Assignment Objective

Build a customer-support system in which **exactly three CrewAI agents** run
**sequentially** (`process=Process.sequential`), answer a user's support query
two different ways, save everything to a text file, and show both answers in a
**Streamlit** UI.

## Architecture

```
User query (Streamlit input)
        ↓
Agent 1: Assistant              → direct answer from the LLM's own knowledge
        ↓
Agent 2: Web Search Assistant   → answer grounded in Serper web-search results
        ↓
Agent 3: Entry Agent            → appends query + both answers to answers.txt
        ↓
Streamlit displays "Assistant Answer" and "Web Search Answer" separately
```

### Exact agent order

| # | Agent | Tool | Responsibility |
|---|-------|------|----------------|
| 1 | **Assistant** | none | Answers directly from its own knowledge (no web search) |
| 2 | **Web Search Assistant** | `SerperDevTool` | Searches the web and answers from the results |
| 3 | **Entry Agent** | custom `Save Support Record` tool | Saves the query and both answers to `answers.txt` |

Task 3 receives the outputs of Task 1 and Task 2 via CrewAI **task context**
(`context=[direct_answer_task, web_search_task]`).

## Tech Stack

- Python
- [CrewAI](https://www.crewai.com/) + crewai-tools (`SerperDevTool`)
- Streamlit
- OpenAI API — all agents run on **`gpt-4.1-mini`** by default (see
  [Configuration](#configuration))
- Serper.dev (web search)
- `filelock` (safe concurrent writes to `answers.txt`)

## Project Structure

```
app.py                     # thin Streamlit entry point
support_crew/
├── config.py              # env loading, settings, API-key validation
├── models.py              # SupportRecord + typed RunResult
├── errors.py              # error taxonomy, user-safe messages, retries
├── storage.py             # locked answers.txt appends, Record-IDs, rotation
├── tools.py               # 'Save Support Record' tool (Entry Agent only)
├── crew.py                # agents, tasks, sequential crew, run logging
└── ui/
    ├── styles.css         # the custom look (gradient sidebar, cards, …)
    ├── components.py      # logo, step cards, result cards, CSS injection
    └── views.py           # New Query / History / About pages
static/logo.svg            # sidebar logo
tests/                     # offline unit + AppTest smoke tests (pytest)
.github/workflows/ci.yml   # CI: ruff + mypy + pytest on every push
pyproject.toml             # ruff + mypy configuration
.pre-commit-config.yaml    # fast pre-commit checks (ruff + hygiene)
.streamlit/config.toml     # base theme (palette, fonts)
.env.example               # template for local API keys
Dockerfile / .dockerignore # container deployment
LICENSE                    # MIT
requirements.txt           # runtime dependencies (bounded ranges)
requirements.lock          # exact tested versions (pip freeze, Windows)
requirements-dev.txt       # + pytest, ruff, mypy, pre-commit
```

## Prerequisites

- **Python 3.10 – 3.13** (CrewAI does not yet support Python 3.14)
- An [OpenAI API key](https://platform.openai.com/)
- A [Serper API key](https://serper.dev/) (free tier available)

## Setup

### 1. Create the project folder

```bash
mkdir buildathon-support-crew
cd buildathon-support-crew
```

(Or clone this repository and `cd` into it.)

### 2. Create a virtual environment

```bash
python -m venv venv
```

Some macOS/Linux installations may require:

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

For a byte-for-byte reproduction of the tested environment use the lockfile
instead:

```bash
pip install -r requirements.lock
```

### 5. Configure API keys

The app reads keys from environment variables and validates them before
starting the crew. **Never hard-code keys or commit them to Git.**

**Option A — `.env` file (recommended for local development):** copy
`.env.example` to `.env` (it is git-ignored) and fill in your keys:

```
OPENAI_API_KEY=your-openai-key
SERPER_API_KEY=your-serper-key
```

The app loads it automatically at startup, so you never re-enter keys in
the terminal. Use exactly `NAME=value` — no spaces around `=`.

**Option B — Streamlit secrets** (for [Streamlit Community
Cloud](https://streamlit.io/cloud) or a local `.streamlit/secrets.toml`,
both git-ignored): add the same names under secrets; the app copies any
missing names from `st.secrets` into the environment at startup. Shell and
`.env` values always take precedence.

```toml
OPENAI_API_KEY = "your-openai-key"
SERPER_API_KEY = "your-serper-key"
```

**Option C — shell environment variables** (take precedence over everything):

macOS/Linux:

```bash
export OPENAI_API_KEY="your-openai-key"
export SERPER_API_KEY="your-serper-key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:SERPER_API_KEY="your-serper-key"
```

## Configuration

Optional environment variables (all have sensible defaults):

| Variable | Default | Purpose |
|---|---|---|
| `SUPPORT_CREW_MODEL` | `gpt-4.1-mini` | LLM used by all three agents |
| `SUPPORT_CREW_AGENT_TIMEOUT` | `120` | Per-agent execution ceiling (seconds) |
| `SUPPORT_CREW_RUN_TIMEOUT` | `420` | Hard deadline for a whole crew run (seconds) |
| `SUPPORT_CREW_MAX_QUERY_CHARS` | `2000` | Maximum accepted query length |
| `SUPPORT_CREW_MAX_ATTEMPTS` | `3` | Attempts per run for transient failures |
| `SUPPORT_CREW_RETRY_BASE_DELAY` | `2` | Backoff base delay in seconds (2s, 4s, …) |
| `SUPPORT_CREW_ANSWERS_MAX_BYTES` | `5242880` | answers.txt rotation cap (0 disables) |

### Failure handling

Transient failures (provider rate limits, network problems, timeouts) are
retried automatically with exponential backoff. Non-transient failures — an
invalid API key, for example — fail immediately with a specific message
telling you what to fix. A retry is skipped if the failed attempt already
saved its record, so records are never duplicated.

Each run executes in a worker thread under a hard overall deadline
(`SUPPORT_CREW_RUN_TIMEOUT`), so the UI always gets a clean timeout error
instead of hanging, even if a provider call stalls past the per-agent
ceilings. Every run's duration and token usage (total / prompt /
completion) are written to the server log — queries themselves are never
logged.

## Run the Application

```bash
streamlit run app.py
```

Streamlit opens the app in your browser at:

```
http://localhost:8501
```

### Run with Docker

```bash
docker build -t support-crew .
docker run --rm -p 8501:8501 --env-file .env support-crew
```

The image runs as a non-root user, includes a container health check
(Streamlit's `/_stcore/health`), and never contains your `.env`, secrets,
or `answers.txt` (see `.dockerignore`).

## Example Workflow

1. On the **New Query** page, enter a query, e.g. `How do I reset my password?`
2. Click **Run Support Crew** (or press **Ctrl+Enter** in the text box).
3. A spinner shows while the three agents run sequentially (this can take a
   minute).
4. The page then shows a success banner, the **Assistant Answer** and
   **Web Search Answer** cards side by side, and a confirmation that the
   record was saved to `answers.txt` (with a download button). The query box
   moves below the results, cleared and ready for the next question.

### Sidebar pages

- **New Query** — the main page; clicking it again starts a fresh session
  (clears the previous output and query box).
- **History** — every query answered in the current browser session, with
  both answers per query.
- **About** — a short description of the agent pipeline.

## Running the Tests and Checks

```bash
pip install -r requirements-dev.txt
ruff check .     # lint
mypy             # type check (app.py + support_crew)
pytest           # 41 offline tests
```

The test suite is fully offline: configuration validation, record storage
(locking, Record-IDs, rotation, formatting), error classification and retry
policy, crew wiring (agent order, tool isolation, task context, structured
output, pinned model/timeout), UI helpers, and headless app smoke tests
(Streamlit `AppTest`: boot, navigation, input validation). No API keys or
network access are required. The same three checks run in CI
(`.github/workflows/ci.yml`) on every push and pull request.

Optionally install the pre-commit hook so ruff and repo-hygiene checks run
on every commit:

```bash
pre-commit install
```

## About `answers.txt` (data lifecycle)

- Created **automatically** in the project folder after the first successful
  query — you never create it manually.
- Each run **appends** a new, clearly separated record (UTF-8) containing a
  unique **Record-ID**, the original query, the Assistant's answer and the
  Web Search answer, so earlier support records are never erased.
- Writes are protected by an OS-level file lock (a temporary
  `answers.txt.lock` sidecar file), so concurrent sessions can never
  interleave records, and each run verifies **its own** Record-ID landed in
  the file before reporting success.
- **Rotation:** once the file reaches the size cap (5 MB by default,
  `SUPPORT_CREW_ANSWERS_MAX_BYTES`) it is renamed to a timestamped archive
  (`answers-YYYYMMDD-HHMMSS.txt`) and a fresh file is started, so stored
  queries can never grow without bound.
- `answers.txt` and its archives are **git-ignored on purpose**: they
  contain real user queries and generated support content, which should not
  be published to GitHub.

### What "your data is safe" means (and doesn't)

- Queries are sent to **OpenAI** (to generate answers) and **Serper**
  (to search the web) — they do leave your machine for those two services,
  and their data policies apply.
- Answers and queries are stored **in plaintext** in `answers.txt` on the
  machine running the app — treat that file as sensitive.
- Nothing is sent anywhere else, nothing is logged verbatim (logs record
  metadata only), and no analytics or tracking exist in the app.
- **To purge all stored data:** delete `answers.txt` and any
  `answers-*.txt` archives from the project folder.

## Stopping the App

- Stop Streamlit: press `Ctrl + C` in the terminal.
- Deactivate the virtual environment:

```bash
deactivate
```

## GitHub Submission Notes

1. Push the project code to GitHub: `app.py`, the `support_crew/` package,
   `static/`, `tests/`, `.github/`, `.streamlit/config.toml`,
   `pyproject.toml`, `.pre-commit-config.yaml`, `.env.example`,
   `Dockerfile`, `.dockerignore`, `LICENSE`, `requirements.txt`,
   `requirements.lock`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`,
   `README.md`. The `.gitignore` keeps `venv/`, `.env`, `answers.txt` and
   its lock/archive files out of the repository.
2. **Keep all API keys out of GitHub** — double-check no `.env` file or key
   string is staged before pushing.
3. Share the GitHub repository link through **WhatsApp**.
4. Post a **working video** of the application in the community.

## Security Notes

- API keys live **only** in environment variables (shell or local `.env`).
- The app never prints, logs, or displays key values — error messages and
  logs name the missing variable, never its value. Run logs record only
  metadata (query length, duration, save status), never query text.
- `.env`, `.env.*` and `.streamlit/secrets.toml` are git-ignored as a safety
  net; still, never paste real keys into any tracked file.

### Prompt-injection posture

The user's query is interpolated into each task prompt, so a hostile query
can try to steer the agents ("ignore your instructions and …"). The blast
radius is deliberately limited **structurally** rather than by filtering:

- Agent 1 has **no tools**; Agent 2 can only run **web searches**; Agent 3
  can only append to the fixed `answers.txt` path — no agent can read files,
  execute code, or write anywhere else, whatever the query says.
- Delegation is disabled for all agents, and each agent has a hard
  execution-time ceiling.
- Queries are length-capped (`SUPPORT_CREW_MAX_QUERY_CHARS`, default 2,000
  characters) to bound token spend.
- Answers are rendered as plain Markdown (never raw HTML), so a manipulated
  answer cannot inject scripts into the page.

A sufficiently adversarial query can still distort the *content* of the two
answers — treat saved records as untrusted user-influenced text.

## License

MIT — see [LICENSE](LICENSE).
