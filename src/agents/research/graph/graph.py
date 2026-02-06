# -*- coding: utf-8 -*-
"""
Research Pipeline Graph
========================

Builds and compiles LangGraph StateGraph for research workflows.

The graph implements a three-phase research pipeline:
1. Planning: rephrase → decompose → initialize_queue
2. Researching: select_next_block → research_block → mark_complete (loop)
3. Reporting: generate_report → save_results

Usage:
    from src.agents.research.graph import build_research_graph

    graph = build_research_graph()
    result = await graph.ainvoke(
        {
            "topic": "Machine Learning Fundamentals",
            "research_id": "research_123",
            "kb_name": "ai_textbook",
            "config": {...},
        },
        config={"configurable": {"thread_id": "research_123"}},
    )
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    check_has_pending,
    decompose_topic,
    generate_report,
    initialize_queue,
    mark_block_complete,
    rephrase_topic,
    research_block,
    save_results,
    select_next_block,
)
from .state import ResearchGraphState


def build_research_graph(
    *,
    enable_checkpoint: bool = False,
) -> Any:
    """
    Build and compile the research pipeline graph.

    The graph follows this flow:

    Phase 1 (Planning):
        START → rephrase_topic → decompose_topic → initialize_queue

    Phase 2 (Researching loop):
        → select_next_block → [has_pending?]
            → yes: research_block → mark_block_complete → select_next_block
            → no:  (continue to Phase 3)

    Phase 3 (Reporting):
        → generate_report → save_results → END

    Args:
        enable_checkpoint: Whether to enable in-memory checkpointing.

    Returns:
        Compiled LangGraph graph ready for ainvoke().
    """
    workflow = StateGraph(ResearchGraphState)

    # --- Add Nodes ---
    # Phase 1: Planning
    workflow.add_node("rephrase_topic", rephrase_topic)
    workflow.add_node("decompose_topic", decompose_topic)
    workflow.add_node("initialize_queue", initialize_queue)

    # Phase 2: Researching
    workflow.add_node("select_next_block", select_next_block)
    workflow.add_node("research_block", research_block)
    workflow.add_node("mark_block_complete", mark_block_complete)

    # Phase 3: Reporting
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("save_results", save_results)

    # --- Edges ---

    # Phase 1: Planning flow
    workflow.add_edge(START, "rephrase_topic")
    workflow.add_edge("rephrase_topic", "decompose_topic")
    workflow.add_edge("decompose_topic", "initialize_queue")
    workflow.add_edge("initialize_queue", "select_next_block")

    # Phase 2: Research loop
    workflow.add_conditional_edges(
        "select_next_block",
        check_has_pending,
        {
            "has_pending": "research_block",
            "all_complete": "generate_report",
        },
    )
    workflow.add_edge("research_block", "mark_block_complete")
    workflow.add_edge("mark_block_complete", "select_next_block")

    # Phase 3: Reporting flow
    workflow.add_edge("generate_report", "save_results")
    workflow.add_edge("save_results", END)

    # --- Compile ---
    checkpointer = None
    if enable_checkpoint:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        except Exception:
            checkpointer = None

    return workflow.compile(checkpointer=checkpointer)


__all__ = ["build_research_graph"]
