# -*- coding: utf-8 -*-
"""
LangChain LLM Provider
======================

Provides LangChain-based LLM integration with:
- Multi-provider support (OpenAI, Anthropic, Ollama, etc.)
- SQLite caching (replaces LightRAG's openai_complete_if_cache)
- Compatible with existing factory interface

Usage:
    from src.services.llm.langchain_provider import LangChainProvider

    # Complete a prompt
    response = await LangChainProvider.complete(
        prompt="Hello",
        system_prompt="You are helpful",
        model="gpt-4o",
        api_key="sk-...",
        base_url="https://api.openai.com/v1",
        binding="openai"
    )

    # Stream a response
    async for chunk in LangChainProvider.stream(...):
        print(chunk, end="")
"""

import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.logging import get_logger

from .capabilities import (
    get_effective_temperature,
    has_thinking_tags,
    supports_response_format,
    system_in_messages,
)
from .config import get_token_limit_kwargs
from .exceptions import LLMAPIError, LLMAuthenticationError, LLMConfigError
from .utils import clean_thinking_tags, is_local_llm_server, sanitize_url

logger = get_logger("LangChain")

# LangChain imports - lazy loaded to handle optional dependency
_langchain_available: Optional[bool] = None

# Langfuse callback handler - lazy loaded
_langfuse_handler: Optional[Any] = None
_langfuse_checked: bool = False


def _get_langfuse_handler():
    """Lazily initialize and return the Langfuse callback handler."""
    global _langfuse_handler, _langfuse_checked
    if _langfuse_checked:
        return _langfuse_handler
    _langfuse_checked = True
    try:
        from langfuse.langchain import CallbackHandler

        _langfuse_handler = CallbackHandler()
        logger.info("Langfuse tracing enabled")
    except Exception:
        _langfuse_handler = None
    return _langfuse_handler


def _check_langchain_available() -> bool:
    """Check if LangChain packages are available."""
    global _langchain_available
    if _langchain_available is None:
        try:
            import langchain_core  # noqa: F401

            _langchain_available = True
        except ImportError:
            _langchain_available = False
    return _langchain_available


class LangChainProvider:
    """
    LangChain-based LLM provider with caching support.

    This provider offers:
    - Multi-provider support through LangChain's unified interface
    - SQLite caching for response deduplication
    - Compatible API with existing factory.complete() interface
    """

    _cache_initialized: bool = False
    _cache_path: str = ".cache/llm_cache.db"

    @classmethod
    def init_cache(cls, cache_path: Optional[str] = None) -> None:
        """
        Initialize SQLite cache for LLM responses.

        Args:
            cache_path: Path to SQLite cache file. Defaults to .cache/llm_cache.db
        """
        if cls._cache_initialized:
            return

        if not _check_langchain_available():
            logger.warning("LangChain not available, caching disabled")
            return

        try:
            from langchain_community.cache import SQLiteCache
            from langchain_core.globals import set_llm_cache

            path = cache_path or cls._cache_path
            cls._cache_path = path

            # Ensure cache directory exists
            cache_dir = Path(path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)

            set_llm_cache(SQLiteCache(database_path=path))
            cls._cache_initialized = True
            logger.info(f"LangChain cache initialized: {path}")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain cache: {e}")

    @classmethod
    def _get_llm(
        cls,
        binding: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Get a LangChain LLM instance for the specified provider.

        Args:
            binding: Provider binding (openai, anthropic, ollama, etc.)
            model: Model name
            api_key: API key
            base_url: Base URL for the API
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            LangChain BaseChatModel instance
        """
        if not _check_langchain_available():
            raise LLMConfigError("LangChain packages not installed")

        binding_lower = (binding or "openai").lower()

        # Apply effective temperature (handles reasoning models like o1)
        effective_temp = get_effective_temperature(binding_lower, model, temperature)

        # Common kwargs for all providers
        common_kwargs: Dict[str, Any] = {
            "temperature": effective_temp,
        }

        if max_tokens:
            common_kwargs["max_tokens"] = max_tokens

        # Route to appropriate provider
        if binding_lower in ["anthropic", "claude"]:
            return cls._get_anthropic_llm(model, api_key, base_url, **common_kwargs, **kwargs)
        elif binding_lower == "ollama" or (base_url and is_local_llm_server(base_url)):
            return cls._get_ollama_llm(model, base_url, **common_kwargs, **kwargs)
        else:
            # Default to OpenAI-compatible
            return cls._get_openai_llm(model, api_key, base_url, **common_kwargs, **kwargs)

    @classmethod
    def _get_openai_llm(
        cls,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Get OpenAI-compatible LLM instance."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise LLMConfigError(
                "langchain-openai not installed. Run: pip install langchain-openai"
            )

        # Sanitize URL
        if base_url:
            base_url = sanitize_url(base_url, model)

        # Use environment variable if api_key not provided
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            **kwargs,
        }

        if api_key:
            llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url

        return ChatOpenAI(**llm_kwargs)

    @classmethod
    def _get_anthropic_llm(
        cls,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Get Anthropic LLM instance."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise LLMConfigError(
                "langchain-anthropic not installed. Run: pip install langchain-anthropic"
            )

        # Use environment variable if api_key not provided
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise LLMAuthenticationError("Anthropic API key not provided", provider="anthropic")

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            **kwargs,
        }

        if base_url:
            llm_kwargs["base_url"] = base_url

        return ChatAnthropic(**llm_kwargs)

    @classmethod
    def _get_ollama_llm(
        cls,
        model: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Get Ollama LLM instance."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fallback to OpenAI-compatible mode for Ollama
            logger.debug("langchain-ollama not installed, using OpenAI-compatible mode")
            return cls._get_openai_llm(model, api_key=None, base_url=base_url, **kwargs)

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            **kwargs,
        }

        if base_url:
            # Ollama base URL should not have /v1 suffix
            base_url = base_url.rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            llm_kwargs["base_url"] = base_url

        return ChatOllama(**llm_kwargs)

    @classmethod
    def _build_messages(
        cls,
        prompt: str,
        system_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        binding: str = "openai",
        model: Optional[str] = None,
    ) -> List[Any]:
        """
        Build LangChain message list from prompt and system prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            messages: Pre-built messages (optional)
            binding: Provider binding for format detection
            model: Model name

        Returns:
            List of LangChain message objects
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if messages:
            # Convert dict messages to LangChain format
            result = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    result.append(SystemMessage(content=content))
                elif role == "assistant":
                    result.append(AIMessage(content=content))
                else:
                    result.append(HumanMessage(content=content))
            return result

        # Build from prompt and system_prompt
        result = []

        # Check if provider supports system in messages
        if system_in_messages(binding, model):
            result.append(SystemMessage(content=system_prompt))

        result.append(HumanMessage(content=prompt))

        return result

    @classmethod
    async def complete(
        cls,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        binding: str = "openai",
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Complete a prompt using LangChain.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            api_key: API key
            base_url: Base URL for the API
            api_version: API version (for Azure OpenAI)
            binding: Provider binding (openai, anthropic, ollama)
            messages: Pre-built messages array (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            response_format: Response format (e.g., {"type": "json_object"})
            **kwargs: Additional arguments

        Returns:
            Generated response text
        """
        if not _check_langchain_available():
            raise LLMConfigError(
                "LangChain packages not installed. Run: pip install langchain-openai"
            )

        # Initialize cache on first call
        cls.init_cache()

        # Prepare kwargs for LLM
        llm_kwargs: Dict[str, Any] = {}

        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens:
            # Get correct token parameter name based on model
            token_kwargs = get_token_limit_kwargs(model or "", max_tokens)
            llm_kwargs.update(token_kwargs)

        # Handle response_format if supported
        binding_lower = (binding or "openai").lower()
        if response_format and supports_response_format(binding_lower, model):
            # LangChain OpenAI supports response_format through model_kwargs
            llm_kwargs["model_kwargs"] = {"response_format": response_format}

        try:
            # Get LLM instance
            llm = cls._get_llm(
                binding=binding,
                model=model or "gpt-4o",
                api_key=api_key,
                base_url=base_url,
                **llm_kwargs,
            )

            # Build messages
            msg_list = cls._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                binding=binding,
                model=model,
            )

            # For Anthropic, pass system separately
            invoke_kwargs: Dict[str, Any] = {}
            if binding_lower in ["anthropic", "claude"] and not messages:
                # Anthropic handles system prompt differently
                # LangChain's ChatAnthropic handles this internally
                pass

            # Invoke LLM with Langfuse tracing
            langfuse_cb = _get_langfuse_handler()
            invoke_config = {"callbacks": [langfuse_cb]} if langfuse_cb else {}
            response = await llm.ainvoke(msg_list, config=invoke_config, **invoke_kwargs)

            # Extract content
            content = response.content if hasattr(response, "content") else str(response)

            # Clean thinking tags if needed
            content = clean_thinking_tags(content, binding, model)

            return content

        except Exception as e:
            error_msg = str(e)

            # Map common errors
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise LLMAuthenticationError(
                    f"Authentication failed: {error_msg}",
                    provider=binding,
                )
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                from .exceptions import LLMRateLimitError

                raise LLMRateLimitError(
                    f"Rate limit exceeded: {error_msg}",
                    provider=binding,
                )
            else:
                raise LLMAPIError(
                    f"LangChain API error: {error_msg}",
                    provider=binding,
                )

    @classmethod
    async def stream(
        cls,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        binding: str = "openai",
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response using LangChain.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            api_key: API key
            base_url: Base URL for the API
            api_version: API version (for Azure OpenAI)
            binding: Provider binding
            messages: Pre-built messages array
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        if not _check_langchain_available():
            raise LLMConfigError(
                "LangChain packages not installed. Run: pip install langchain-openai"
            )

        # Prepare kwargs for LLM
        llm_kwargs: Dict[str, Any] = {}

        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens:
            token_kwargs = get_token_limit_kwargs(model or "", max_tokens)
            llm_kwargs.update(token_kwargs)

        try:
            # Get LLM instance
            llm = cls._get_llm(
                binding=binding,
                model=model or "gpt-4o",
                api_key=api_key,
                base_url=base_url,
                **llm_kwargs,
            )

            # Build messages
            msg_list = cls._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                binding=binding,
                model=model,
            )

            # Track thinking block state for streaming
            binding_lower = (binding or "openai").lower()
            should_clean_thinking = has_thinking_tags(binding_lower, model)
            in_thinking_block = False
            thinking_buffer = ""

            # Stream response with Langfuse tracing
            langfuse_cb = _get_langfuse_handler()
            stream_config = {"callbacks": [langfuse_cb]} if langfuse_cb else {}
            async for chunk in llm.astream(msg_list, config=stream_config):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)

                if not content:
                    continue

                # Handle thinking tags in streaming
                if should_clean_thinking:
                    if "<think>" in content:
                        in_thinking_block = True
                        thinking_buffer = content
                        continue
                    elif in_thinking_block:
                        thinking_buffer += content
                        if "</think>" in thinking_buffer:
                            # End of thinking block, clean and yield
                            cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                            if cleaned:
                                yield cleaned
                            in_thinking_block = False
                            thinking_buffer = ""
                        continue

                yield content

        except Exception as e:
            error_msg = str(e)

            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise LLMAuthenticationError(
                    f"Authentication failed: {error_msg}",
                    provider=binding,
                )
            else:
                raise LLMAPIError(
                    f"LangChain stream error: {error_msg}",
                    provider=binding,
                )


__all__ = [
    "LangChainProvider",
]
