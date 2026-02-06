# -*- coding: utf-8 -*-
"""
Research Graph State Definitions
=================================

TypedDict state schemas for LangGraph research workflows.
"""

from typing import Any, TypedDict


class ResearchGraphState(TypedDict, total=False):
    """State for the research pipeline graph."""

    # --- Inputs ---
    topic: str
    research_id: str
    kb_name: str
    config: dict[str, Any]

    # --- Phase 1: Planning ---
    optimized_topic: str
    sub_topics: list[dict[str, Any]]  # From DecomposeAgent

    # --- Phase 2: Researching ---
    pending_blocks: list[str]  # Block IDs waiting to be processed
    completed_blocks: list[str]  # Block IDs already completed
    current_block_id: str | None  # Currently processing block
    queue_data: dict[str, Any]  # Serialized queue state

    # --- Phase 3: Reporting ---
    report_result: dict[str, Any]  # From ReportingAgent

    # --- Final Output ---
    result: dict[str, Any]
    error: str | None


__all__ = ["ResearchGraphState"]
