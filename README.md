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
- OpenAI API (LLM)
- Serper.dev (web search)

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

### 5. Configure API keys

The app reads keys from environment variables and validates them before
starting the crew. **Never hard-code keys or commit them to Git.**

**Option A — `.env` file (recommended for local development):** create a
file named `.env` in the project folder (it is git-ignored) with:

```
OPENAI_API_KEY=your-openai-key
SERPER_API_KEY=your-serper-key
```

The app loads it automatically at startup, so you never re-enter keys in
the terminal. Use exactly `NAME=value` — no spaces around `=`.

**Option B — shell environment variables** (take precedence over `.env`):

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

## Run the Application

```bash
streamlit run app.py
```

Streamlit opens the app in your browser at:

```
http://localhost:8501
```

## Example Workflow

1. Enter a query, e.g. `How do I reset my password?`
2. Click **Run Support Crew**.
3. Watch the status messages:
   - Assistant is preparing a direct answer…
   - Web Search Assistant is researching the query…
   - Entry Agent is saving the result…
4. The page shows **Assistant Answer** and **Web Search Answer** in separate
   sections, plus a confirmation that the record was saved to `answers.txt`.

## About `answers.txt`

- Created **automatically** in the project folder after the first successful
  query — you never create it manually.
- Each run **appends** a new, clearly separated record (UTF-8) containing the
  original query, the Assistant's answer and the Web Search answer, so earlier
  support records are never erased.
- `answers.txt` is **git-ignored on purpose**: it contains real user queries
  and generated support content, which should not be published to GitHub.

## Stopping the App

- Stop Streamlit: press `Ctrl + C` in the terminal.
- Deactivate the virtual environment:

```bash
deactivate
```

## GitHub Submission Notes

1. Push the project code to GitHub (only `app.py`, `requirements.txt`,
   `.gitignore`, `README.md` — the `.gitignore` keeps `venv/`, `.env` and
   `answers.txt` out of the repository).
2. **Keep all API keys out of GitHub** — double-check no `.env` file or key
   string is staged before pushing.
3. Share the GitHub repository link through **WhatsApp**.
4. Post a **working video** of the application in the community.

## Security Reminder

- API keys live **only** in your shell environment variables.
- The app never prints, logs, or displays key values — error messages name
  the missing variable, never its value.
- `.env`, `.env.*` and `.streamlit/secrets.toml` are git-ignored as a safety
  net; still, never paste real keys into any tracked file.
