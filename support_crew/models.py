"""Shared data models."""

from dataclasses import dataclass

from pydantic import BaseModel


class SupportRecord(BaseModel):
    """Reliably parseable final output returned by the Entry Agent."""

    assistant_answer: str
    web_search_answer: str
    file_saved: bool
    file_path: str


@dataclass
class RunResult:
    """Typed outcome of one crew run; stored in session state and history."""

    query: str
    assistant_answer: str
    web_search_answer: str
    file_saved: bool
    file_path: str
    completed_at: float | None = None
    # Token accounting (from CrewAI's usage metrics; None when unavailable).
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
