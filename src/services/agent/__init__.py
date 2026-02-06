# -*- coding: utf-8 -*-
"""
Agent Services
==============

Services for agent infrastructure:
- AgentConfigResolver: Configuration resolution with priority
- LLMOrchestrator: LLM call orchestration with logging and tracking

Usage:
    from src.services.agent import AgentConfigResolver, LLMOrchestrator

    # Create config resolver
    config = AgentConfigResolver("research", "research_agent")

    # Create orchestrator
    orchestrator = LLMOrchestrator(
        config_resolver=config,
        agent_name="research_agent",
        module_name="research",
    )

    # Call LLM
    response = await orchestrator.complete(
        user_prompt="Hello",
        system_prompt="You are helpful",
    )
"""

from .config_resolver import AgentConfigResolver, ResolvedAgentConfig
from .orchestrator import LLMOrchestrator

__all__ = [
    "AgentConfigResolver",
    "ResolvedAgentConfig",
    "LLMOrchestrator",
]
