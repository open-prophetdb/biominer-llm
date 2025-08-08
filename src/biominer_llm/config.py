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
        env_prefix="BIOMINER_AI_LLM_", env_file=".env", env_nested_delimiter="__"
    )

    @model_validator(mode="before")
    def adjust_llm_config(cls, data: dict):
        allowed_providers = ["openai", "anthropic", "ollama", "xai", "gemini"]
        provider = data.get("provider", None)
        api_key = data.get("api_key", None)
        if data.get("base_url", None) is None:
            if data.get("provider", None) not in allowed_providers:
                raise ValueError(
                    f"Unsupported provider: {data.get('provider', None)} or set base_url"
                )
        else:
            if data.get("provider", None) in allowed_providers:
                data["provider"] = "custom"
                provider = "custom"

        if api_key is None and provider in allowed_providers:
            if data.get(f"{provider}_api_key") is not None:
                data["api_key"] = data.get(f"{provider}_api_key")
            elif os.environ[f"{provider.upper()}_API_KEY"] is not None:
                data["api_key"] = data.get(f"{provider.upper()}_API_KEY")
            else:
                raise ValueError(f"Please set api_key for {provider}")
        elif api_key is None:
            raise ValueError("Please set api_key")
        elif api_key is not None and data.get("provider", None) in allowed_providers:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key

        known = set(cls.model_fields.keys())
        data = {k: v for k, v in data.items() if k in known}
        return data
