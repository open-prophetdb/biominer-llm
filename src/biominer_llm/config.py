from typing import Optional
from pydantic import model_validator
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class LLMConfig(BaseSettings):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: Optional[float] = 0.7
    max_tokens: int = 4096
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="BIOMINER_AI_LLM_",
        env_file=".env",
        env_nested_delimiter="__",
    )

    @model_validator(mode="before")
    def adjust_llm_config(self, data):
        allowed_providers = ["openai", "anthropic", "ollama", "xai", "gemini"]
        if data["base_url"] is None:
            if data["provider"] not in allowed_providers:
                raise ValueError(
                    f"Unsupported provider: {data['provider']} or set base_url"
                )

        from dotenv import dotenv_values

        env = dotenv_values(".env")
        if data["api_key"] is None and data["provider"] in allowed_providers:
            if env.get(f"{data['provider'].upper()}_API_KEY") is None:
                raise ValueError("Please set api_key")
            else:
                data["api_key"] = env.get(f"{data['provider'].upper()}_API_KEY")
        elif data["api_key"] is None:
            raise ValueError("Please set api_key")
        elif data["api_key"] is not None and data["provider"] in allowed_providers:
            os.environ[f"{data['provider'].upper()}_API_KEY"] = data["api_key"]
