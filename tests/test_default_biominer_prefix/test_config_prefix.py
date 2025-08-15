import os
import pytest
from biominer_llm.config import LLMConfig


@pytest.fixture(autouse=True, scope="session")
def set_working_dir():
    old = os.getcwd()
    import pathlib

    os.chdir(pathlib.Path(__file__).parent)
    yield
    os.chdir(old)


def test_llmconfig(monkeypatch):
    monkeypatch.setenv("BIOMINER_AI_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("BIOMINER_AI_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("BIOMINER_AI_LLM_API_KEY", "anthropic_dummy-api-key")
    monkeypatch.setenv("BIOMINER_AI_LLM_TEMPERATURE", "1")
    monkeypatch.setenv("BIOMINER_AI_LLM_MAX_TOKENS", "1024")

    config = LLMConfig()
    assert config.provider == "anthropic"
    assert config.model == "gpt-4o"
    assert config.temperature == 1
    assert config.max_tokens == 1024
    assert config.api_key.get_secret_value() == "anthropic_dummy-api-key"

    monkeypatch.delenv("BIOMINER_AI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("BIOMINER_AI_LLM_MODEL", raising=False)
    monkeypatch.delenv("BIOMINER_AI_LLM_API_KEY", raising=False)
    monkeypatch.delenv("BIOMINER_AI_LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("BIOMINER_AI_LLM_MAX_TOKENS", raising=False)
