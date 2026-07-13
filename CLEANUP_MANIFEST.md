# Cleanup Manifest

Audit date: 2026-07-12. Every deleted, moved, renamed, or materially
modified file from the structure audit is recorded here. Application source
code was **not** modified by this audit; behavior is unchanged.

| Item | Original Location | Final Location | Action | Reason | Risk | Validation |
|------|-------------------|----------------|--------|--------|------|------------|
| `app.cpython-312.pyc` | `__pycache__/` | — | Deleted | Stale bytecode of the pre-refactor monolithic app.py; regenerable; git-ignored | None | App boots; 44/44 tests pass |
| `app.cpython-314.pyc` | `__pycache__/` | — | Deleted | Bytecode from an unsupported Python 3.14 run; project targets 3.10–3.13 | None | App boots; 44/44 tests pass |
| `__pycache__/` (root) | project root | — | Deleted (dir) | Emptied by the two deletions above; Python recreates on demand | None | App boots; 44/44 tests pass |
| `.gitignore` | project root | project root | Modified | Added `!.env.example` (was silently ignored by `.env.*`) and OS metadata patterns (`.DS_Store`, `Thumbs.db`) | Low | `git check-ignore` confirms `.env.example` trackable and `.env` still ignored |
| `.env.example` | project root (untracked) | project root (tracked) | Added to git | README instructs users to copy it, but the repo never contained it | None | `git ls-files` shows it tracked; placeholders only, no real values |
| `PROJECT_AUDIT_REPORT.md` | — | project root | Created | Required audit deliverable | None | — |
| `CLEANUP_MANIFEST.md` | — | project root | Created | Required audit deliverable | None | — |

## Explicitly preserved (not safe to remove automatically)

| Item | Why preserved |
|------|---------------|
| `.env` | Real credentials; user-owned; untracked + ignored (verified never in git history) |
| `answers.txt` | User data (queries + answers); untracked + ignored; rotation-capped |
| `venv/` | Local environment; ignored; deleting would break the running app |
| `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` | Functional tool caches; ignored; no repo impact |
| `.claude/launch.json` | Actively used by the dev harness to launch the app; flagged for manual decision (see audit report §10) |
| `requirements.lock` | Reproducibility record of the tested (Windows) environment; Linux equivalent produced by CI as an artifact |

## Not done, deliberately

- **No files moved or renamed** — the layout already follows Python/Streamlit
  conventions; restructuring would be cosmetic and add risk for zero benefit.
- **No dependencies removed** — declared dependencies match actual imports 1:1
  (verified by grep of all `import`/`from` statements against
  `requirements*.txt`).
- **No dead code removed** — none exists (ruff F-rules clean; all modules
  reachable from entry points or tests; no commented-out legacy code).
