#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified BaseAgent - Simplified base class for all module agents.

This is the single source of truth for agent base functionality across:
- solve module
- research module
- guide module
- ideagen module
- co_writer module
- question module

Refactored to delegate to dedicated services:
- AgentConfigResolver: Configuration management
- LLMOrchestrator: LLM call orchestration
- PromptManager: Prompt loading
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from src.logging import get_logger
from src.services.agent import AgentConfigResolver, LLMOrchestrator
from src.services.prompt import get_prompt_manager


class BaseAgent(ABC):
    """
    Simplified base class for all module agents.

    This class provides:
    - LLM configuration management (via AgentConfigResolver)
    - Agent parameters (temperature, max_tokens) from agents.yaml
    - Prompt loading via PromptManager
    - Unified LLM call interface (via LLMOrchestrator)
    - Token tracking (via LLMOrchestrator)
    - Logging

    Subclasses must implement the `process()` method.

    Example:
        class MyAgent(BaseAgent):
            def __init__(self, config=None):
                super().__init__(
                    module_name="mymodule",
                    agent_name="my_agent",
                    config=config,
                )

            async def process(self, query: str) -> str:
                response = await self.call_llm(
                    user_prompt=query,
                    system_prompt=self.get_prompt("system"),
                )
                return response
    """

    def __init__(
        self,
        module_name: str,
        agent_name: str,
        language: str = "zh",
        config: Optional[dict[str, Any]] = None,
        token_tracker: Any = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize base Agent.

        Args:
            module_name: Module name (solve/research/guide/ideagen/co_writer/question)
            agent_name: Agent name (e.g., "solve_agent", "note_agent")
            language: Language setting ('zh' | 'en'), default 'zh'
            config: Optional configuration dictionary
            token_tracker: Optional external TokenTracker instance
            log_dir: Optional log directory path
        """
        self.module_name = module_name
        self.agent_name = agent_name
        self.language = language
        self.config = config or {}

        # Initialize logger
        logger_name = f"{module_name.capitalize()}.{agent_name}"
        self.logger = get_logger(logger_name, log_dir=log_dir)

        # Initialize config resolver (handles all LLM and agent parameter resolution)
        self._config_resolver = AgentConfigResolver(module_name, agent_name)

        # Initialize LLM orchestrator (handles LLM calls, logging, tracking)
        self._orchestrator = LLMOrchestrator(
            config_resolver=self._config_resolver,
            agent_name=agent_name,
            module_name=module_name,
            logger=self.logger,
            token_tracker=token_tracker,
        )

        # Load prompts using unified PromptManager
        try:
            self.prompts = get_prompt_manager().load_prompts(
                module_name=module_name,
                agent_name=agent_name,
                language=language,
            )
            if self.prompts:
                self.logger.debug(f"Prompts loaded: {agent_name} ({language})")
        except Exception as e:
            self.prompts = None
            self.logger.warning(f"Failed to load prompts for {agent_name}: {e}")

        # Agent status from config
        self.agent_config = self.config.get("agents", {}).get(agent_name, {})
        self.enabled = self.agent_config.get("enabled", True)

    # -------------------------------------------------------------------------
    # Backward-compatible properties (delegate to ConfigResolver)
    # -------------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Get current model name (backward compatibility)."""
        return self._config_resolver.get_model()

    @property
    def base_url(self) -> Optional[str]:
        """Get current base URL (backward compatibility)."""
        return self._config_resolver.get_base_url()

    @property
    def api_key(self) -> Optional[str]:
        """Get current API key (backward compatibility)."""
        return self._config_resolver.get_api_key()

    # -------------------------------------------------------------------------
    # Configuration Access (delegates to ConfigResolver)
    # -------------------------------------------------------------------------

    def get_model(self, override: Optional[str] = None) -> str:
        """
        Get model name with priority resolution.

        Args:
            override: Optional explicit model override

        Returns:
            Model name

        Raises:
            ValueError: If model is not configured
        """
        return self._config_resolver.get_model(override)

    def get_temperature(self, override: Optional[float] = None) -> float:
        """
        Get temperature parameter from unified config (agents.yaml).

        Args:
            override: Optional explicit temperature override

        Returns:
            Temperature value
        """
        return self._config_resolver.get_temperature(override)

    def get_max_tokens(self, override: Optional[int] = None) -> int:
        """
        Get maximum token count from unified config (agents.yaml).

        Args:
            override: Optional explicit max_tokens override

        Returns:
            Maximum token count
        """
        return self._config_resolver.get_max_tokens(override)

    def get_max_retries(self) -> int:
        """
        Get maximum retry count.

        Returns:
            Retry count
        """
        return self._config_resolver.get_max_retries()

    def refresh_config(self) -> None:
        """
        Refresh LLM configuration from the current active settings.

        This method reloads the LLM configuration from the unified config service,
        allowing agents to pick up configuration changes made by users in Settings
        without needing to restart the server or recreate the agent instance.
        """
        self._config_resolver.refresh()
        self.logger.debug("Configuration refreshed")

    # -------------------------------------------------------------------------
    # LLM Call Interface (delegates to LLMOrchestrator)
    # -------------------------------------------------------------------------

    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        messages: Optional[list[dict[str, str]]] = None,
        response_format: Optional[dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        verbose: bool = True,
        stage: Optional[str] = None,
    ) -> str:
        """
        Unified interface for calling LLM (non-streaming).

        Uses the LLM factory to route calls to the appropriate provider
        (cloud or local) based on configuration.

        Args:
            user_prompt: User prompt (ignored if messages provided)
            system_prompt: System prompt (ignored if messages provided)
            messages: Pre-built messages array (optional, overrides prompt/system_prompt)
            response_format: Response format (e.g., {"type": "json_object"})
            temperature: Temperature parameter (optional, uses config by default)
            max_tokens: Maximum tokens (optional, uses config by default)
            model: Model name (optional, uses config by default)
            verbose: Whether to print raw LLM output (default True)
            stage: Stage marker for logging and tracking

        Returns:
            LLM response text
        """
        return await self._orchestrator.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stage=stage,
            verbose=verbose,
        )

    async def stream_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        messages: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Unified interface for streaming LLM responses.

        Uses the LLM factory to route calls to the appropriate provider
        (cloud or local) based on configuration.

        Args:
            user_prompt: User prompt (ignored if messages provided)
            system_prompt: System prompt (ignored if messages provided)
            messages: Pre-built messages array (optional, overrides prompt/system_prompt)
            temperature: Temperature parameter (optional, uses config by default)
            max_tokens: Maximum tokens (optional, uses config by default)
            model: Model name (optional, uses config by default)
            stage: Stage marker for logging

        Yields:
            Response chunks as strings
        """
        async for chunk in self._orchestrator.stream(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stage=stage,
        ):
            yield chunk

    # -------------------------------------------------------------------------
    # Token Tracking & Statistics (delegates to LLMOrchestrator)
    # -------------------------------------------------------------------------

    @classmethod
    def get_stats(cls, module_name: str):
        """
        Get shared LLMStats tracker for a module.

        Args:
            module_name: Module name

        Returns:
            LLMStats instance
        """
        return LLMOrchestrator.get_stats(module_name)

    @classmethod
    def reset_stats(cls, module_name: Optional[str] = None):
        """
        Reset shared stats.

        Args:
            module_name: Module name (if None, reset all)
        """
        LLMOrchestrator.reset_stats(module_name)

    @classmethod
    def print_stats(cls, module_name: Optional[str] = None):
        """
        Print stats summary.

        Args:
            module_name: Module name (if None, print all)
        """
        LLMOrchestrator.print_stats(module_name)

    # -------------------------------------------------------------------------
    # Prompt Helpers
    # -------------------------------------------------------------------------

    def get_prompt(
        self,
        section_or_type: str = "system",
        field_or_fallback: Optional[str] = None,
        fallback: str = "",
    ) -> Optional[str]:
        """
        Get prompt by type or section/field.

        Supports two calling patterns:
        1. get_prompt("system") - simple key lookup
        2. get_prompt("section", "field", "fallback") - nested lookup (for research module)

        Args:
            section_or_type: Prompt type key or section name
            field_or_fallback: Field name (if nested) or fallback value (if simple)
            fallback: Fallback value if prompt not found (only used in nested mode)

        Returns:
            Prompt string or fallback
        """
        if not self.prompts:
            return (
                fallback
                if fallback
                else (
                    field_or_fallback
                    if isinstance(field_or_fallback, str) and field_or_fallback
                    else None
                )
            )

        # Check if this is a nested lookup (section.field pattern)
        section_value = self.prompts.get(section_or_type)

        if isinstance(section_value, dict) and field_or_fallback is not None:
            # Nested lookup: get_prompt("section", "field", "fallback")
            result = section_value.get(field_or_fallback)
            if result is not None:
                return result
            return fallback if fallback else None
        else:
            # Simple lookup: get_prompt("key") or get_prompt("key", "fallback")
            if section_value is not None:
                return section_value
            # field_or_fallback acts as fallback in simple mode
            return field_or_fallback if field_or_fallback else (fallback if fallback else None)

    def has_prompts(self) -> bool:
        """Check if prompts have been loaded."""
        return self.prompts is not None

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """
        Check if Agent is enabled.

        Returns:
            Whether enabled
        """
        return self.enabled

    # -------------------------------------------------------------------------
    # Abstract Method
    # -------------------------------------------------------------------------

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """
        Main processing logic of Agent (must be implemented by subclasses).

        Returns:
            Processing result
        """

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation of Agent."""
        return (
            f"{self.__class__.__name__}("
            f"module={self.module_name}, "
            f"name={self.agent_name}, "
            f"enabled={self.enabled})"
        )


__all__ = ["BaseAgent"]
