from typing import Optional, Dict, Any
from pydantic import model_validator, field_validator
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
        extra="ignore",
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

        # If api_key is not set, try to get it from environment variables
        if api_key is None and provider in allowed_providers:
            # First try to get from provider-specific environment variable
            provider_env_key = f"{provider.upper()}_API_KEY"
            if provider_env_key in os.environ and os.environ[provider_env_key]:
                data["api_key"] = SecretStr(os.environ[provider_env_key])
            else:
                # If not found, try to get from common environment variables
                common_env_keys = [
                    "XAI_API_KEY",
                    "OPENAI_API_KEY", 
                    "ANTHROPIC_API_KEY",
                    "OLLAMA_API_KEY",
                    "GEMINI_API_KEY"
                ]
                
                found_key = None
                for env_key in common_env_keys:
                    if env_key in os.environ and os.environ[env_key]:
                        found_key = os.environ[env_key]
                        break
                
                if found_key:
                    data["api_key"] = SecretStr(found_key)
                else:
                    raise ValueError(f"Please set api_key for {provider}. You can set BIOMINER_AI_LLM_API_KEY, {provider_env_key}, or any of the common API keys: {', '.join(common_env_keys)}")
        elif api_key is None:
            raise ValueError("Please set api_key")
        elif api_key is not None and data.get("provider", None) in allowed_providers:
            # Set the provider-specific environment variable if api_key is provided
            provider_env_key = f"{provider.upper()}_API_KEY"
            if isinstance(api_key, SecretStr):
                os.environ[provider_env_key] = api_key.get_secret_value()
            else:
                os.environ[provider_env_key] = str(api_key)

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
