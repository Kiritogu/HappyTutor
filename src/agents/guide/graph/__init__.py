# -*- coding: utf-8 -*-
"""
Guide Learning Graph
=====================

LangGraph-based orchestration for guided learning workflows.

Usage:
    from src.agents.guide.graph import (
        build_session_graph,
        build_interaction_graph,
        GuideGraphState,
        CreateSessionState,
    )

    # Create a learning session
    session_graph = build_session_graph()
    result = await session_graph.ainvoke(
        {"notebook_id": "nb_123", "notebook_name": "My Notebook", "records": [...]},
        config={"configurable": {"language": "en"}},
    )

    # Handle user interaction
    interaction_graph = build_interaction_graph(enable_checkpoint=True)
    result = await interaction_graph.ainvoke(
        {"action": "start", "session_id": result["session_id"], ...},
        config={"configurable": {"thread_id": result["session_id"]}},
    )
"""

from .graph import build_interaction_graph, build_session_graph
from .state import CreateSessionState, GuideGraphState

__all__ = [
    "build_session_graph",
    "build_interaction_graph",
    "GuideGraphState",
    "CreateSessionState",
]
