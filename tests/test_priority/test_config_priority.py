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


def test_llmconfig_manual():
    config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="dummy",
        temperature=0.8,
        max_tokens=1024,
    )
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 1024
    assert config.api_key.get_secret_value() == "dummy"


def test_llmconfig_env_file():
    config = LLMConfig(_env_file=".env")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 4096
    assert config.api_key.get_secret_value() == "BIOMINER_AI_LLM_API_KEY"


def test_llmconfig_biominer_prefix_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_API_KEY")
    monkeypatch.setenv("BIOMINER_AI_LLM_API_KEY", "BIOMINER_AI_LLM_API_KEY_dummy")
    config = LLMConfig(_env_file=".1env")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 4096
    assert config.api_key.get_secret_value() == "BIOMINER_AI_LLM_API_KEY_dummy"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BIOMINER_AI_LLM_API_KEY", raising=False)


def test_llmconfig_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_API_KEY")
    config = LLMConfig(_env_file=".1env")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 4096
    assert config.api_key.get_secret_value() == "OPENAI_API_KEY"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
