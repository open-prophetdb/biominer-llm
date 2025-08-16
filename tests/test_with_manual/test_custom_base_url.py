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


def test_llmconfig_custom_base_url():
    config = LLMConfig(base_url="http://localhost", api_key="dummy")
    assert config.provider == "custom"

    config = LLMConfig(provider="openai", base_url="http://localhost", api_key="dummy")
    assert config.provider == "openai"

    config = LLMConfig(
        provider="biominer", base_url="http://localhost", api_key="dummy"
    )
    assert config.provider == "biominer"
