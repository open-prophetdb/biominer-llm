import logging
from typing import Optional, Dict, Any
from pydantic import model_validator, field_validator
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class LLMConfig(BaseSettings):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: Optional[float] = 0.7
    max_tokens: int = 4096
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="BIOMINER_AI_LLM_",
        env_file=os.getenv("BIOMINER_AI_ENV", ".env"),
        env_nested_delimiter="__",
        extra="ignore",
    )

    @model_validator(mode="before")
    def adjust_llm_config(cls, data: dict):
        allowed_providers = ["openai", "anthropic", "ollama", "xai", "gemini"]
        provider = data.get("provider", None)
        api_key = data.get("api_key", None)
        if data.get("base_url", None) is None:
            if data.get("provider", "openai") not in allowed_providers:
                raise ValueError(
                    f"Unsupported provider: {data.get('provider', None)} or set base_url"
                )
        else:
            if provider in allowed_providers or provider is None:
                logger.warning(
                    "You specified a base_url, we guess you are using a custom provider, so we will set provider to custom."
                )
                data["provider"] = "custom"

        if api_key is None and provider in allowed_providers:
            if data.get(f"{provider}_api_key") is not None:
                logger.warning(
                    f"We didn't find api_key in the config, but we found {provider}_api_key in the config, so we will use it."
                )
                data["api_key"] = data.get(f"{provider}_api_key")
            elif os.environ.get(f"{provider.upper()}_API_KEY", None) is not None:
                logger.warning(
                    f"We didn't find api_key in the config, but we found {provider.upper()}_API_KEY in the environment variables, so we will use it."
                )
                data["api_key"] = data.get(f"{provider.upper()}_API_KEY")
            else:
                raise ValueError(f"Please set api_key for {provider}")
        elif data.get("api_key", None) is None:
            raise ValueError("Please set api_key")

        known = set(cls.model_fields.keys())
        data = {k: v for k, v in data.items() if k in known}
        return data

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("Max tokens must be greater than 0")
        return v

    @classmethod
    def from_env(cls, env_prefix: str = "BIOMINER_AI_LLM_"):
        return cls(**{k: v for k, v in os.environ.items() if k.startswith(env_prefix)})

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)
