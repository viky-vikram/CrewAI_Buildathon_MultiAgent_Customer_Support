"""missing_api_keys: names-only validation of required environment variables."""

from support_crew import config


def test_no_keys_missing_when_both_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    assert config.missing_api_keys() == []


def test_reports_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    assert config.missing_api_keys() == ["OPENAI_API_KEY"]


def test_reports_all_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert config.missing_api_keys() == ["OPENAI_API_KEY", "SERPER_API_KEY"]


def test_blank_or_whitespace_value_counts_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    assert config.missing_api_keys() == ["OPENAI_API_KEY"]
