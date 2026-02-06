# -*- coding: utf-8 -*-
"""
Application Warmup Service
==========================

Pre-initializes expensive resources at startup to reduce first-request latency.

Warmup targets:
1. LLM configuration and client
2. HTTP connection pool (via warmup LLM call)
3. Tiktoken encoding
4. Prompt templates
5. Common agent instances
"""

import asyncio
import time
from typing import Optional

from src.logging import get_logger

logger = get_logger("Warmup")


async def warmup_llm_connection(timeout: float = 30.0) -> bool:
    """
    Warm up LLM connection by making a minimal API call.

    This establishes the HTTP connection pool and validates credentials.

    Args:
        timeout: Maximum time to wait for warmup (seconds)

    Returns:
        True if warmup succeeded, False otherwise
    """
    try:
        from src.services.llm import complete as llm_complete, get_llm_config

        config = get_llm_config()
        if not config or not config.api_key:
            logger.warning("LLM not configured, skipping connection warmup")
            return False

        logger.info(f"Warming up LLM connection (model={config.model})...")
        start = time.time()

        # Make a minimal LLM call to establish connection
        response = await asyncio.wait_for(
            llm_complete(
                prompt="Hi",
                system_prompt="Reply with just 'ok'",
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                max_tokens=5,
                temperature=0,
            ),
            timeout=timeout,
        )

        elapsed = time.time() - start
        logger.success(f"LLM connection warmed up in {elapsed:.2f}s")
        return True

    except asyncio.TimeoutError:
        logger.warning(f"LLM warmup timed out after {timeout}s")
        return False
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")
        return False


def warmup_tiktoken() -> bool:
    """
    Pre-load tiktoken encoding to avoid first-call latency.

    Returns:
        True if warmup succeeded, False otherwise
    """
    try:
        import tiktoken

        logger.info("Loading tiktoken encoding...")
        start = time.time()

        # Load the most commonly used encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        # Warm up the encoder with a sample text
        _ = encoding.encode("Hello, this is a warmup text for tiktoken.")

        elapsed = time.time() - start
        logger.success(f"Tiktoken loaded in {elapsed:.3f}s")
        return True

    except ImportError:
        logger.debug("Tiktoken not installed, skipping warmup")
        return False
    except Exception as e:
        logger.warning(f"Tiktoken warmup failed: {e}")
        return False


def warmup_prompt_manager() -> bool:
    """
    Pre-load prompt manager and common prompt templates.

    Returns:
        True if warmup succeeded, False otherwise
    """
    try:
        from src.services.prompt import get_prompt_manager

        logger.info("Loading prompt templates...")
        start = time.time()

        pm = get_prompt_manager()

        # Pre-load prompts for commonly used agents
        common_agents = [
            ("chat", "chat_agent", "zh"),
            ("chat", "chat_agent", "en"),
            ("research", "research_agent", "zh"),
            ("guide", "chat_agent", "zh"),
        ]

        loaded = 0
        for module, agent, lang in common_agents:
            try:
                prompts = pm.load_prompts(module, agent, lang)
                if prompts:
                    loaded += 1
            except Exception:
                pass

        elapsed = time.time() - start
        logger.success(f"Loaded {loaded} prompt templates in {elapsed:.3f}s")
        return True

    except Exception as e:
        logger.warning(f"Prompt manager warmup failed: {e}")
        return False


def warmup_config_services() -> bool:
    """
    Pre-load configuration services.

    Returns:
        True if warmup succeeded, False otherwise
    """
    try:
        from src.services.config import get_agent_params
        from src.services.llm import get_llm_config

        logger.info("Loading configuration services...")
        start = time.time()

        # Load LLM config
        get_llm_config()

        # Load agent params for common modules
        for module in ["chat", "research", "guide", "question"]:
            try:
                get_agent_params(module)
            except Exception:
                pass

        elapsed = time.time() - start
        logger.success(f"Configuration loaded in {elapsed:.3f}s")
        return True

    except Exception as e:
        logger.warning(f"Config warmup failed: {e}")
        return False


async def warmup_all(
    skip_llm_call: bool = False,
    llm_timeout: float = 30.0,
) -> dict[str, bool]:
    """
    Run all warmup tasks.

    Args:
        skip_llm_call: If True, skip the actual LLM API call (useful for offline mode)
        llm_timeout: Timeout for LLM warmup call

    Returns:
        Dictionary with warmup results for each component
    """
    logger.info("=" * 50)
    logger.info("Starting application warmup...")
    logger.info("=" * 50)

    start = time.time()
    results = {}

    # 1. Config services (sync, fast)
    results["config"] = warmup_config_services()

    # 2. Tiktoken (sync, can be slow first time)
    results["tiktoken"] = warmup_tiktoken()

    # 3. Prompt manager (sync)
    results["prompts"] = warmup_prompt_manager()

    # 4. LLM connection (async, slowest)
    if not skip_llm_call:
        results["llm_connection"] = await warmup_llm_connection(timeout=llm_timeout)
    else:
        logger.info("Skipping LLM connection warmup (offline mode)")
        results["llm_connection"] = None

    elapsed = time.time() - start

    # Summary
    succeeded = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)

    logger.info("=" * 50)
    logger.success(f"Warmup completed: {succeeded}/{total} tasks in {elapsed:.2f}s")
    logger.info("=" * 50)

    return results


__all__ = [
    "warmup_all",
    "warmup_llm_connection",
    "warmup_tiktoken",
    "warmup_prompt_manager",
    "warmup_config_services",
]
