# -*- coding: utf-8 -*-
"""
Guide Learning Graphs
======================

Builds and compiles LangGraph StateGraphs for guided learning workflows.

Two separate graphs:
1. Session creation graph: Analyzes notebook and creates learning plan
2. Interaction graph: Handles user interactions during learning

Usage:
    from src.agents.guide.graph import build_session_graph, build_interaction_graph

    # Create session
    session_graph = build_session_graph()
    session_result = await session_graph.ainvoke(
        {"notebook_id": "...", "notebook_name": "...", "records": [...]},
        config={"configurable": {"language": "en"}},
    )

    # User interaction
    interaction_graph = build_interaction_graph()
    result = await interaction_graph.ainvoke(
        {"action": "start", "session_id": "...", ...},
        config={"configurable": {"thread_id": session_id, "language": "en"}},
    )
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .nodes import (
    advance_index,
    check_learning_complete,
    fix_interactive,
    generate_interactive,
    generate_summary,
    handle_chat,
    locate_knowledge,
    route_action,
)
from .state import CreateSessionState, GuideGraphState


def build_session_graph(
    *,
    enable_checkpoint: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Any:
    """
    Build the session creation graph.

    This graph runs once to analyze notebook content and create a learning plan.

    Flow: START → locate_knowledge → END

    Args:
        enable_checkpoint: Whether to enable SQLite checkpointing.
        checkpoint_path: Path for the checkpoint database.

    Returns:
        Compiled LangGraph graph ready for ainvoke().
    """
    workflow = StateGraph(CreateSessionState)

    # Add the single node
    workflow.add_node("locate_knowledge", locate_knowledge)

    # Simple flow
    workflow.add_edge(START, "locate_knowledge")
    workflow.add_edge("locate_knowledge", END)

    # Compile
    checkpointer = None
    if enable_checkpoint:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = checkpoint_path or Path("data/checkpoints/guide_session.sqlite")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except Exception:
            checkpointer = None

    return workflow.compile(checkpointer=checkpointer)


def build_interaction_graph(
    *,
    enable_checkpoint: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Any:
    """
    Build the interaction graph for user learning sessions.

    This graph handles individual user actions during a learning session.
    Use with checkpointing to persist state across requests.

    Flow:
        START → route_action
          → "start":    generate_interactive → END
          → "next":     advance_index → [complete?]
                            → yes: generate_summary → END
                            → no:  generate_interactive → END
          → "chat":     handle_chat → END
          → "fix_html": fix_interactive → END
          → "invalid":  END

    Args:
        enable_checkpoint: Whether to enable SQLite checkpointing.
        checkpoint_path: Path for the checkpoint database.

    Returns:
        Compiled LangGraph graph ready for ainvoke().
    """
    workflow = StateGraph(GuideGraphState)

    # --- Add Nodes ---
    workflow.add_node("route_node", _passthrough)  # Routing entry point
    workflow.add_node("generate_interactive", generate_interactive)
    workflow.add_node("handle_chat", handle_chat)
    workflow.add_node("fix_interactive", fix_interactive)
    workflow.add_node("advance_index", advance_index)
    workflow.add_node("generate_summary", generate_summary)

    # --- Edges ---

    # START → route_node
    workflow.add_edge(START, "route_node")

    # route_node → conditional routing based on action
    workflow.add_conditional_edges(
        "route_node",
        route_action,
        {
            "start": "generate_interactive",
            "next": "advance_index",
            "chat": "handle_chat",
            "fix_html": "fix_interactive",
            "invalid": END,
        },
    )

    # generate_interactive → END (for start action)
    workflow.add_edge("generate_interactive", END)

    # handle_chat → END
    workflow.add_edge("handle_chat", END)

    # fix_interactive → END
    workflow.add_edge("fix_interactive", END)

    # advance_index → check if complete
    workflow.add_conditional_edges(
        "advance_index",
        check_learning_complete,
        {
            "complete": "generate_summary",
            "continue": "generate_interactive",
        },
    )

    # generate_summary → END
    workflow.add_edge("generate_summary", END)

    # --- Compile ---
    checkpointer = None
    if enable_checkpoint:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = checkpoint_path or Path("data/checkpoints/guide_interaction.sqlite")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except Exception:
            checkpointer = None

    return workflow.compile(checkpointer=checkpointer)


async def _passthrough(state: GuideGraphState, config: RunnableConfig) -> dict:
    """Pass-through node used for routing."""
    return {}


__all__ = ["build_session_graph", "build_interaction_graph"]
