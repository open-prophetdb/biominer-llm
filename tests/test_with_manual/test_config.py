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


def test_llmconfig_provider_validation():
    with pytest.raises(ValidationError):
        LLMConfig(provider="invalid")

    config = LLMConfig(api_key="dummy")
    assert config.provider == "openai"

    config = LLMConfig(provider="anthropic", api_key="dummy")
    assert config.provider == "anthropic"

    config = LLMConfig(base_url="http://localhost", api_key="dummy")
    assert config.provider == "custom"

    config = LLMConfig(
        provider="biominer", base_url="http://localhost", api_key="dummy"
    )
    assert config.provider == "biominer"


def test_llmconfig_api_key_validation(monkeypatch):
    config = LLMConfig(api_key="dummy")
    assert config.provider == "openai"
    assert config.api_key.get_secret_value() == "dummy"

    config = LLMConfig(provider="anthropic", api_key="dummy")
    assert config.provider == "anthropic"
    assert config.api_key.get_secret_value() == "dummy"

    config = LLMConfig(base_url="http://localhost", api_key="dummy")
    assert config.provider == "custom"
    assert config.api_key.get_secret_value() == "dummy"

    with pytest.raises(ValidationError):
        LLMConfig(base_url="http://localhost")

    with pytest.raises(ValidationError):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        LLMConfig()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    print(os.environ["OPENAI_API_KEY"])
    config = LLMConfig()

    assert config.provider == "openai"
    assert config.api_key.get_secret_value() == "dummy"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_llmconfig_temperature_validation():
    config = LLMConfig(api_key="dummy", temperature=0.5)
    assert config.temperature == 0.5

    with pytest.raises(ValidationError):
        LLMConfig(api_key="dummy", temperature=1.5)
    with pytest.raises(ValidationError):
        LLMConfig(api_key="dummy", temperature=-0.1)


def test_llmconfig_max_tokens_validation():
    config = LLMConfig(api_key="dummy", max_tokens=1024)
    assert config.max_tokens == 1024

    with pytest.raises(ValidationError):
        LLMConfig(api_key="dummy", max_tokens=0)
    with pytest.raises(ValidationError):
        LLMConfig(api_key="dummy", max_tokens=-10)


def test_llmconfig_from_dict():
    data = {"provider": "openai", "api_key": "dummy", "model": "gpt-4o"}
    config = LLMConfig.from_dict(data)
    assert config.provider == "openai"
    assert config.api_key.get_secret_value() == "dummy"


def test_llmconfig_to_dict():
    config = LLMConfig(api_key="dummy")
    d = config.to_dict()
    assert d["provider"] == "openai"
    assert d["api_key"] == "dummy" or d["api_key"].get_secret_value() == "dummy"


def test_llmconfig_from_env(monkeypatch):
    monkeypatch.setenv("BIOMINER_AI_LLM_API_KEY", "dummy")
    config = LLMConfig.from_env()
    assert config.api_key.get_secret_value() == "dummy"
