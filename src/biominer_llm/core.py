from typing import Dict, Any, Optional
import logging
from langchain_core.messages import BaseMessage
from .config import LLMConfig

logger = logging.getLogger(__name__)


def init_llm(
    config: Optional[LLMConfig] = None, extra_config: Optional[Dict[str, Any]] = None
):
    """
    Initialize a Large Language Model instance based on the specified provider.

    Args:
        config: LLMConfig object containing provider, model, temperature, max_tokens, api_key, and base_url.
        extra_config: Additional configuration parameters specific to the provider.

    Returns:
        An initialized LLM instance from the specified provider.

    Raises:
        ValueError: If an unsupported provider is specified.
        ImportError: If the required provider package is not installed.

    Example:
        >>> # Initialize OpenAI GPT-4
        >>> llm = init_llm(config=LLMConfig(provider="openai", model="gpt-4o"))

        >>> # Initialize Anthropic Claude with custom config
        >>> llm = init_llm(
        ...     config=LLMConfig(
        ...         provider="anthropic",
        ...         model="claude-3-sonnet-20240229",
        ...         temperature=0.5,
        ...         max_tokens=4096,
        ...     ),
        ...     extra_config={"top_p": 0.9}
        ... )
    """
    logger.info(f"Initializing LLM with config: {config}, it's a {type(config)}")
    # Handle both LLMConfig object and string provider
    if config is None:
        config = LLMConfig()
    if isinstance(config, LLMConfig):
        # It's an LLMConfig object
        provider = config.provider
        llm_model = config.model
        temperature = config.temperature
        max_tokens = config.max_tokens
        base_url = str(config.base_url) if config.base_url else None
        api_key = config.api_key.get_secret_value()
    else:
        raise ValueError("config must be an LLMConfig object.")

    if base_url:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            verbose=True,
            **(extra_config or {}),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            api_key=api_key,
            verbose=True,
            **(extra_config or {}),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model_name=llm_model,
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
            api_key=api_key,
            verbose=True,
            **(extra_config or {}),
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=llm_model,
            temperature=temperature,
            api_key=api_key,
            verbose=True,
            **(extra_config or {}),
        )
    elif provider == "xai":
        from langchain_xai import ChatXAI

        llm = ChatXAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            verbose=True,
            **(extra_config or {}),
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            verbose=True,
            **(extra_config or {}),
        )
    else:
        raise ValueError(f"Please set a supported provider: {provider} or set base_url")

    return llm


def get_tokens_usage(response: BaseMessage) -> Dict[str, int]:
    """Get the number of tokens in the response"""
    if hasattr(response, "usage_metadata") and response.usage_metadata:  # type: ignore
        return response.usage_metadata  # type: ignore

    return {}
