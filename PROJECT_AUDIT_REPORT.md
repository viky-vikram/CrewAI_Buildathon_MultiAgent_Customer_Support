# Project Audit Report

**Project:** Multi-Agent Customer Support System (CrewAI + Streamlit)
**Audit date:** 2026-07-12
**Auditor role:** Senior Software Architect / Production Readiness / Security Review / Refactoring

---

## 1. Original Project Structure Summary (before this audit)

The project had already been refactored from a 905-line single-file app into a
conventional Python package layout in the immediately preceding engineering
rounds (commit `695883c`). This audit therefore found a structurally sound
project and focused on repository hygiene, sensitive data, dependency
accuracy, and validation.

```
app.py                      # Streamlit entry point (thin, 41 lines)
support_crew/               # application package
├── config.py               # settings, env loading, key validation
├── models.py               # SupportRecord, RunResult
├── errors.py               # error taxonomy, retries, run deadline
├── storage.py              # locked answers.txt appends, IDs, rotation
├── tools.py                # Entry Agent file tool
├── crew.py                 # agents/tasks/crew + execution
└── ui/                     # views, components, styles.css
static/logo.svg
tests/                      # 44 offline tests (unit + AppTest smoke)
.github/workflows/ci.yml    # ruff + mypy + pytest + Linux lock artifact
.streamlit/config.toml      # theme
.claude/launch.json         # dev-tool launch config (see §10)
Dockerfile / .dockerignore / LICENSE / pyproject.toml / pytest.ini
requirements.txt / requirements.lock / requirements-dev.txt
.env (local only) / .env.example / answers.txt (local only)
__pycache__/                # stale (see below)   [REMOVED]
.mypy_cache/ .pytest_cache/ .ruff_cache/          [ignored, kept]
venv/                                             [ignored, kept]
```

## 2. Dependency Map (verified before any change)

- `app.py` → `support_crew.ui.{components,views}` → `support_crew.{config,crew,errors,models}` → `support_crew.{tools,storage}` → `support_crew.config`
- No circular imports (mypy + import order verify this; `ui` never imports upward into `crew` beyond the public `run_support_crew`).
- Data files read at runtime: `support_crew/ui/styles.css`, `static/logo.svg`,
  `answers.txt` (+ `.lock` sidecar, timestamped archives).
- Entry points: `streamlit run app.py` (dev/local), Dockerfile CMD (container),
  `.claude/launch.json` (IDE/dev harness).
- Third-party imports ↔ `requirements.txt`: **exact 1:1 match**
  (`streamlit`, `crewai`, `crewai_tools`, `dotenv`→python-dotenv,
  `filelock`, `pydantic`). Dev-only: `pytest` (+ ruff/mypy/pre-commit as
  tools, not imports).

## 3. Problems Identified

| # | Problem | Risk | Resolution |
|---|---------|------|------------|
| 1 | `.env.example` was **silently untracked**: the `.gitignore` pattern `.env.*` matched it, so it never entered git despite being referenced by README and believed committed | Medium (broken onboarding; docs promised a file the repo didn't have) | Added `!.env.example` negation; file now tracked |
| 2 | Stale `__pycache__/` at repo root containing bytecode of the **old monolithic app.py** compiled under Python 3.14 (unsupported) and 3.12 | Low (ignored by git; pure clutter) | Deleted (regenerable) |
| 3 | `.gitignore` lacked OS-metadata patterns (`.DS_Store`, `Thumbs.db`) | Low | Added |
| 4 | `.claude/launch.json` is tracked but absent from README's submission list; contains a Windows-style venv path | Info | Preserved (actively used by the dev harness to launch the app); documented here |
| 5 | `requirements.lock` is Windows-frozen (contains `pywin32`) and cannot drive Linux installs | Info (already mitigated) | Already documented in Dockerfile; CI exports `requirements-linux.lock` artifact per run |

## 4. Sensitive-Data Findings

- `.env` exists locally with real values for `OPENAI_API_KEY=sk-****REDACTED****`
  and `SERPER_API_KEY=****REDACTED****`. It is **untracked and ignored** —
  verified correct.
- **Full git history audited** (`git log --all --name-only`): `.env`,
  `answers.txt`, and secrets have **never been committed** in any revision.
  **No key rotation is required.**
- Pattern scan of every tracked/working file (excluding `venv/`) for
  `sk-…`, AWS `AKIA…`, private-key blocks, GitHub `ghp_…`, and hardcoded
  `api_key/password/token/secret = "<value>"` assignments: **zero matches**.
- `.env.example` contains placeholder names only (`your-openai-key`) —
  cannot be mistaken for real credentials.
- Startup validation reports missing variable **names** (never values);
  logs record run metadata only (query length, duration, tokens), never
  query text or key material.
- `answers.txt` contains real user queries (PII): untracked, ignored,
  size-capped with rotation, purge instructions documented in README.

## 5. Cleanup Classification & Actions

**Safe to remove automatically (removed):**
- `__pycache__/app.cpython-312.pyc`, `__pycache__/app.cpython-314.pyc` —
  stale bytecode of a source layout that no longer exists; regenerable;
  git-ignored. Evidence of non-use: Python regenerates per import; the 3.14
  build cannot even run this project (CrewAI unsupported on 3.14).

**Preserved (ignored caches, functional):**
- `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `venv/` — untracked and
  git-ignored; deleting them would only slow the next tool run. No repo
  hygiene impact.

**Must not remove (preserved):**
- `.env` (real credentials — user-owned), `answers.txt` (user data),
  `.claude/launch.json` (used by dev tooling), `requirements.lock`
  (reproducibility record; Linux variant produced by CI).

**Files/folders moved or renamed: none.** The structure already follows
Streamlit/Python conventions; moving files would violate the "no cosmetic
restructuring" rule.

**Dead code removed: none found.** ruff (F401/F841 etc.) is clean; every
module is imported from an entry point or tests; no commented-out legacy
blocks exist.

**Dependencies removed: none.** Every declared runtime dependency maps to a
real import; every import maps to a declaration. Dev dependencies are
correctly separated in `requirements-dev.txt`.

## 6. Configuration Review

- All tunables are environment variables with safe defaults, centralized in
  `support_crew/config.py`; precedence (shell > .env > st.secrets) is
  implemented and documented.
- No hardcoded secrets, URLs, or environment-specific values in source.
- Timeouts: per-agent (120 s) and per-run (420 s) ceilings; retries with
  exponential backoff; file-size caps for stored data; 2,000-char input cap.
- Port 8501 appears in `Dockerfile` (EXPOSE/CMD — conventional) and the dev
  `launch.json` (dev-only). Streamlit's own config can override; acceptable.
- Debug modes: none enabled; Streamlit runs headless in the container as a
  non-root user with a health check.

## 7. Tests

- 44 tests, all offline: config validation matrix, storage (locking,
  Record-IDs, rotation, unicode), error taxonomy + retry/backoff/veto/
  deadline semantics, crew wiring (order, tool isolation, contexts,
  structured output, pinned model), UI helpers, and 6 headless AppTest
  smoke tests (boot, navigation, validation).
- No tests were added or modified by this audit (none were needed for the
  changes made; the .gitignore change is covered by the tracking check
  below).

## 8. Validation Results

See `CLEANUP_MANIFEST.md` §Validation and the summary table in the final
response. All gates re-run after changes: ruff clean, mypy clean,
44/44 tests pass, app boots and renders correctly (browser-verified),
`.env.example` confirmed tracked, `.env` confirmed still ignored.

## 9. Remaining Risks (honest)

1. **Windows lockfile**: `requirements.lock` reproduces the dev machine, not
   Linux. Mitigated (CI artifact) but the canonical lock is still
   platform-bound. Consider committing the CI-generated Linux lock.
2. **Streamlit DOM coupling**: `styles.css` targets internal test-ids;
   safe only while streamlit stays pinned to 1.59.x (it is).
3. **Synchronous runs**: a crew run occupies a session thread for its
   duration (bounded by the 420 s deadline). Fine at small scale.
4. **Transitive dependency CVEs**: no vulnerability scanner has been run
   against the 159-package tree (pip-audit is not installed). Recommend
   adding `pip-audit` to CI. **Not claimed as passing — not executed.**
5. **`answers.txt` plaintext PII** at rest (documented, rotated, capped) —
   acceptable for the assignment; use SQLite + encryption if this grows.

## 10. Files Requiring Manual Review

- `.claude/launch.json` — keep tracked if the team uses the same dev
  tooling; untrack (`git rm --cached`) if you consider it personal IDE
  configuration. It is harmless either way; README's submission list
  intentionally omits it.

## 11. Recommended Future Improvements

- Add `pip-audit` (dependency CVE scan) and a secret-scanning action
  (e.g. gitleaks) to CI.
- Commit the CI-produced `requirements-linux.lock` for reproducible Docker
  builds.
- Async crew execution with live per-agent progress if the UI constraint is
  ever relaxed; SQLite storage for true multi-user deployments.

## 12. Before / After Folder Tree

**Before:** identical to §1 including stale `__pycache__/` and an untracked
`.env.example`.

**After:** identical minus `__pycache__/`, plus `.env.example` actually in
git, plus `PROJECT_AUDIT_REPORT.md` and `CLEANUP_MANIFEST.md`. No file was
moved or renamed. Application behavior is byte-identical.
