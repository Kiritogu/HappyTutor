# -*- coding: utf-8 -*-
"""
Agent Configuration Resolver
============================

Resolves agent configuration with proper priority:
1. Explicit overrides (passed to methods)
2. Agent-specific config (from agents.yaml)
3. LLM config (from unified config service)
4. Environment variables
5. Defaults
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

from src.services.config import get_agent_params
from src.services.llm import get_llm_config, get_token_limit_kwargs, supports_response_format


@dataclass
class ResolvedAgentConfig:
    """Resolved configuration for an agent."""

    # LLM settings
    model: str
    api_key: str
    base_url: Optional[str]
    api_version: Optional[str]
    binding: str
    # Agent parameters
    temperature: float
    max_tokens: int
    max_retries: int


class AgentConfigResolver:
    """
    Resolves agent configuration from multiple sources.

    This service centralizes configuration resolution logic that was
    previously scattered across BaseAgent.__init__() and various getter methods.

    Priority for each parameter:
    1. Explicit overrides (passed to methods)
    2. Agent-specific config (from agents.yaml)
    3. LLM config (from unified config service)
    4. Environment variables
    5. Defaults

    Usage:
        resolver = AgentConfigResolver("research", "research_agent")
        model = resolver.get_model()  # Resolves with priority
        kwargs = resolver.get_llm_kwargs(temperature=0.5)  # Build kwargs for LLM call
    """

    def __init__(self, module_name: str, agent_name: str):
        """
        Initialize the config resolver.

        Args:
            module_name: Module name (e.g., "research", "guide", "chat")
            agent_name: Agent name (e.g., "research_agent", "chat_agent")
        """
        self.module_name = module_name
        self.agent_name = agent_name
        self._agent_params = get_agent_params(module_name)
        self._llm_config = None
        self._refresh_llm_config()

    def _refresh_llm_config(self) -> None:
        """Refresh LLM config from unified config service."""
        try:
            self._llm_config = get_llm_config()
        except Exception:
            self._llm_config = None

    @property
    def llm_config(self):
        """Get the current LLM config (read-only access)."""
        return self._llm_config

    def get_model(self, override: Optional[str] = None) -> str:
        """
        Get model name with priority resolution.

        Priority:
        1. override parameter
        2. LLM config from unified service
        3. LLM_MODEL environment variable

        Args:
            override: Optional explicit model override

        Returns:
            Model name

        Raises:
            ValueError: If no model is configured
        """
        if override:
            return override
        if self._llm_config and self._llm_config.model:
            return self._llm_config.model
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            return env_model
        raise ValueError(
            f"Model not configured for agent {self.agent_name}. "
            "Please set LLM_MODEL in .env or configure a provider."
        )

    def get_temperature(self, override: Optional[float] = None) -> float:
        """
        Get temperature parameter.

        Priority:
        1. override parameter
        2. Agent params from agents.yaml

        Args:
            override: Optional explicit temperature override

        Returns:
            Temperature value
        """
        if override is not None:
            return override
        return self._agent_params["temperature"]

    def get_max_tokens(self, override: Optional[int] = None) -> int:
        """
        Get max_tokens parameter.

        Priority:
        1. override parameter
        2. Agent params from agents.yaml

        Args:
            override: Optional explicit max_tokens override

        Returns:
            Max tokens value
        """
        if override is not None:
            return override
        return self._agent_params["max_tokens"]

    def get_max_retries(self) -> int:
        """
        Get max_retries from settings.

        Returns:
            Max retries value
        """
        from src.config.settings import settings

        return settings.retry.max_retries

    def get_api_key(self) -> Optional[str]:
        """Get API key from LLM config."""
        if self._llm_config:
            return self._llm_config.api_key
        return os.getenv("LLM_API_KEY")

    def get_base_url(self) -> Optional[str]:
        """Get base URL from LLM config."""
        if self._llm_config:
            return self._llm_config.base_url
        return os.getenv("LLM_HOST")

    def get_api_version(self) -> Optional[str]:
        """Get API version from LLM config (for Azure OpenAI)."""
        if self._llm_config:
            return getattr(self._llm_config, "api_version", None)
        return os.getenv("LLM_API_VERSION")

    def get_binding(self) -> str:
        """Get provider binding from LLM config."""
        if self._llm_config:
            return getattr(self._llm_config, "binding", "openai")
        return os.getenv("LLM_BINDING", "openai")

    def get_llm_kwargs(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Build kwargs dict for LLM calls.

        This method handles:
        - Token limit naming (max_tokens vs max_completion_tokens for newer models)
        - Response format validation based on provider capabilities

        Args:
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Dictionary of kwargs ready for LLM call
        """
        resolved_model = self.get_model(model)
        resolved_temp = self.get_temperature(temperature)
        resolved_max_tokens = self.get_max_tokens(max_tokens)

        kwargs: dict[str, Any] = {"temperature": resolved_temp}

        # Handle token limit for newer models (o1, gpt-4o, etc.)
        if resolved_max_tokens:
            kwargs.update(get_token_limit_kwargs(resolved_model, resolved_max_tokens))

        # Handle response_format with capability check
        if response_format:
            binding = self.get_binding()
            if supports_response_format(binding, resolved_model):
                kwargs["response_format"] = response_format

        return kwargs

    def refresh(self) -> None:
        """
        Refresh configuration from services.

        Call this when you need to pick up configuration changes
        made by users in Settings without restarting the server.
        """
        self._agent_params = get_agent_params(self.module_name)
        self._refresh_llm_config()

    def resolve_all(self) -> ResolvedAgentConfig:
        """
        Resolve all configuration into a single dataclass.

        Returns:
            ResolvedAgentConfig with all resolved values
        """
        return ResolvedAgentConfig(
            model=self.get_model(),
            api_key=self.get_api_key() or "",
            base_url=self.get_base_url(),
            api_version=self.get_api_version(),
            binding=self.get_binding(),
            temperature=self.get_temperature(),
            max_tokens=self.get_max_tokens(),
            max_retries=self.get_max_retries(),
        )


__all__ = [
    "AgentConfigResolver",
    "ResolvedAgentConfig",
]
