# -*- coding: utf-8 -*-
"""
LLM Orchestrator
================

Orchestrates LLM calls with:
- Parameter resolution via AgentConfigResolver
- Logging (input/output)
- Token usage tracking
- Error handling with context

This service extracts the LLM call orchestration logic that was
previously in BaseAgent.call_llm() and BaseAgent.stream_llm().
"""

import time
from typing import Any, AsyncGenerator, Optional

from src.logging import LLMStats, get_logger
from src.services.llm import complete as llm_complete
from src.services.llm import stream as llm_stream

from .config_resolver import AgentConfigResolver


class LLMOrchestrator:
    """
    Orchestrates LLM calls for agents.

    This class centralizes all LLM call orchestration logic:
    - Parameter resolution (via AgentConfigResolver)
    - Automatic logging of inputs and outputs
    - Token usage tracking (external tracker + shared LLMStats)
    - Error handling with context

    Usage:
        config = AgentConfigResolver("research", "research_agent")
        orchestrator = LLMOrchestrator(config, "research_agent", "research")

        # Non-streaming call
        response = await orchestrator.complete(
            user_prompt="Hello",
            system_prompt="You are helpful",
        )

        # Streaming call
        async for chunk in orchestrator.stream(
            user_prompt="Hello",
            system_prompt="You are helpful",
        ):
            print(chunk, end="")
    """

    # Shared stats per module (class-level singleton pattern)
    _stats: dict[str, LLMStats] = {}

    def __init__(
        self,
        config_resolver: AgentConfigResolver,
        agent_name: str,
        module_name: str,
        logger: Any = None,
        token_tracker: Any = None,
    ):
        """
        Initialize the LLM orchestrator.

        Args:
            config_resolver: AgentConfigResolver instance for parameter resolution
            agent_name: Agent name for logging and tracking
            module_name: Module name for stats grouping
            logger: Optional custom logger (defaults to module.agent logger)
            token_tracker: Optional external token tracker instance
        """
        self.config = config_resolver
        self.agent_name = agent_name
        self.module_name = module_name
        self.logger = logger or get_logger(f"{module_name}.{agent_name}")
        self.token_tracker = token_tracker

    @classmethod
    def get_stats(cls, module_name: str) -> LLMStats:
        """
        Get or create shared LLMStats for a module.

        Args:
            module_name: Module name

        Returns:
            LLMStats instance for the module
        """
        if module_name not in cls._stats:
            cls._stats[module_name] = LLMStats(module_name=module_name.capitalize())
        return cls._stats[module_name]

    @classmethod
    def reset_stats(cls, module_name: Optional[str] = None) -> None:
        """
        Reset stats for a module or all modules.

        Args:
            module_name: Module name (if None, reset all)
        """
        if module_name:
            if module_name in cls._stats:
                cls._stats[module_name].reset()
        else:
            for stats in cls._stats.values():
                stats.reset()

    @classmethod
    def print_stats(cls, module_name: Optional[str] = None) -> None:
        """
        Print stats summary.

        Args:
            module_name: Module name (if None, print all)
        """
        if module_name:
            if module_name in cls._stats:
                cls._stats[module_name].print_summary()
        else:
            for stats in cls._stats.values():
                stats.print_summary()

    def _track_tokens(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        stage: str,
    ) -> None:
        """
        Track token usage.

        Records usage to both:
        1. External TokenTracker (if provided)
        2. Shared LLMStats (always)

        Args:
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            response: LLM response text
            stage: Stage name for tracking
        """
        # External tracker (if provided)
        if self.token_tracker:
            try:
                self.token_tracker.add_usage(
                    agent_name=self.agent_name,
                    stage=stage,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_text=response,
                )
            except Exception:
                pass  # Don't let tracking errors affect main flow

        # Shared stats (always)
        stats = self.get_stats(self.module_name)
        stats.add_call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
        )

    async def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        messages: Optional[list[dict[str, str]]] = None,
        response_format: Optional[dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        stage: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        """
        Complete a prompt with full orchestration.

        Handles:
        - Parameter resolution via ConfigResolver
        - Input/output logging
        - Token tracking
        - Error handling

        Args:
            user_prompt: User prompt (ignored if messages provided)
            system_prompt: System prompt (ignored if messages provided)
            messages: Pre-built messages array (optional)
            response_format: Response format (e.g., {"type": "json_object"})
            temperature: Temperature override
            max_tokens: Max tokens override
            model: Model override
            stage: Stage name for logging/tracking
            verbose: Whether to log output

        Returns:
            LLM response text
        """
        resolved_model = self.config.get_model(model)
        stage_label = stage or self.agent_name

        # Build kwargs
        kwargs = self.config.get_llm_kwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        if messages:
            kwargs["messages"] = messages

        # Log input
        start_time = time.time()
        if hasattr(self.logger, "log_llm_input"):
            self.logger.log_llm_input(
                agent_name=self.agent_name,
                stage=stage_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={"model": resolved_model, **kwargs},
            )

        # Call LLM via factory
        try:
            response = await llm_complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                api_key=self.config.get_api_key(),
                base_url=self.config.get_base_url(),
                api_version=self.config.get_api_version(),
                max_retries=self.config.get_max_retries(),
                **kwargs,
            )
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

        # Calculate duration
        duration = time.time() - start_time

        # Track tokens
        self._track_tokens(resolved_model, system_prompt, user_prompt, response, stage_label)

        # Log output
        if hasattr(self.logger, "log_llm_output"):
            self.logger.log_llm_output(
                agent_name=self.agent_name,
                stage=stage_label,
                response=response,
                metadata={"length": len(response), "duration": duration},
            )

        if verbose:
            self.logger.debug(f"LLM response: model={resolved_model}, duration={duration:.2f}s")

        return response

    async def stream(
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
        Stream a response with full orchestration.

        Handles:
        - Parameter resolution via ConfigResolver
        - Input/output logging
        - Token tracking (after stream completes)
        - Error handling

        Args:
            user_prompt: User prompt (ignored if messages provided)
            system_prompt: System prompt (ignored if messages provided)
            messages: Pre-built messages array (optional)
            temperature: Temperature override
            max_tokens: Max tokens override
            model: Model override
            stage: Stage name for logging/tracking

        Yields:
            Response chunks as strings
        """
        resolved_model = self.config.get_model(model)
        stage_label = stage or self.agent_name

        # Build kwargs
        kwargs = self.config.get_llm_kwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Log input
        start_time = time.time()
        if hasattr(self.logger, "log_llm_input"):
            self.logger.log_llm_input(
                agent_name=self.agent_name,
                stage=stage_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={"model": resolved_model, "streaming": True},
            )

        full_response = ""

        try:
            async for chunk in llm_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=resolved_model,
                api_key=self.config.get_api_key(),
                base_url=self.config.get_base_url(),
                api_version=self.config.get_api_version(),
                messages=messages,
                **kwargs,
            ):
                full_response += chunk
                yield chunk

            # Track tokens after streaming completes
            self._track_tokens(
                resolved_model, system_prompt, user_prompt, full_response, stage_label
            )

            # Log output
            duration = time.time() - start_time
            if hasattr(self.logger, "log_llm_output"):
                self.logger.log_llm_output(
                    agent_name=self.agent_name,
                    stage=stage_label,
                    response=full_response[:200] + "..."
                    if len(full_response) > 200
                    else full_response,
                    metadata={
                        "length": len(full_response),
                        "duration": duration,
                        "streaming": True,
                    },
                )

        except Exception as e:
            self.logger.error(f"LLM streaming failed: {e}")
            raise


__all__ = [
    "LLMOrchestrator",
]
