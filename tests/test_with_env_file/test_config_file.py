import os
import pytest
from pydantic import ValidationError
from biominer_llm.config import LLMConfig


@pytest.fixture(autouse=True, scope="session")
def set_working_dir():
    old = os.getcwd()
    import pathlib

    os.chdir(pathlib.Path(__file__).parent)
    yield
    os.chdir(old)


def test_llmconfig():
    config = LLMConfig(_env_file=".1env")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 4096
    assert config.api_key.get_secret_value() == "BIOMINER_AI_LLM_API_KEY"


def test_llmconfig_openai():
    config = LLMConfig(_env_file=".2env")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 4096
    assert config.api_key.get_secret_value() == "your_openai_api_key_here"


def test_llmconfig_invalid_env_file(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        LLMConfig(_env_file=".3env")


def test_llmconfig_custom_provider(monkeypatch):
    monkeypatch.setenv("BIOMINER_AI_ENV", ".5env")
    import importlib
    import biominer_llm.config

    importlib.reload(biominer_llm.config)
    config = biominer_llm.config.LLMConfig()
    assert config.provider == "custom"
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 4096
    assert config.api_key.get_secret_value() == "dummy"
    assert config.base_url == "http://localhost"
