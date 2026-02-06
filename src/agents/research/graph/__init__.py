# -*- coding: utf-8 -*-
"""
Research Pipeline Graph
========================

LangGraph-based orchestration for research workflows.

Usage:
    from src.agents.research.graph import build_research_graph, ResearchGraphState

    graph = build_research_graph()
    result = await graph.ainvoke(
        {
            "topic": "Machine Learning Fundamentals",
            "research_id": "research_123",
            "kb_name": "ai_textbook",
            "config": {...},
        },
        config={"configurable": {"thread_id": "research_123", "ws_callback": callback}},
    )
"""

from .graph import build_research_graph
from .state import ResearchGraphState

__all__ = [
    "build_research_graph",
    "ResearchGraphState",
]
