"""Crew wiring: agent order, tool isolation, task context, structured output.

These tests construct the crew but never kick it off — no network calls.
"""

import pytest
from crewai import Process

import app
from app import SupportRecord, build_crew


@pytest.fixture()
def crew_parts(monkeypatch):
    # SerperDevTool reads its key at construction; a dummy value keeps the
    # wiring test offline and independent of the developer's real .env.
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    return build_crew()


def test_exactly_three_agents_in_required_order(crew_parts):
    crew, *_ = crew_parts
    roles = [agent.role for agent in crew.agents]
    assert roles == ["Assistant", "Web Search Assistant", "Entry Agent"]


def test_process_is_sequential_with_three_tasks(crew_parts):
    crew, task1, task2, task3 = crew_parts
    assert crew.process == Process.sequential
    assert list(crew.tasks) == [task1, task2, task3]


def test_tool_isolation_per_agent(crew_parts):
    crew, *_ = crew_parts
    assistant, web_search_assistant, entry_agent = crew.agents
    assert assistant.tools == []
    assert [type(t).__name__ for t in web_search_assistant.tools] == ["SerperDevTool"]
    assert [t.name for t in entry_agent.tools] == ["Save Support Record"]


def test_task2_context_is_explicitly_empty(crew_parts):
    _, _, task2, _ = crew_parts
    assert task2.context == []


def test_task3_receives_both_answers_as_context(crew_parts):
    _, task1, task2, task3 = crew_parts
    assert task3.context == [task1, task2]


def test_task3_returns_structured_support_record(crew_parts):
    *_, task3 = crew_parts
    assert task3.output_pydantic is SupportRecord


def test_agents_use_pinned_model_and_timeout(crew_parts):
    crew, *_ = crew_parts
    for agent in crew.agents:
        assert app.MODEL_NAME in str(agent.llm.model)
        assert agent.max_execution_time == app.AGENT_MAX_EXECUTION_TIME
        assert agent.allow_delegation is False
