# -*- coding: utf-8 -*-
"""
Question Generation Graph
==========================

Builds and compiles LangGraph StateGraph for question generation workflows.

Two modes:
1. Single question: retrieve → generate_single → END
2. Custom (multi-question): retrieve → plan → generate_and_analyze (loop) → build_summary → END

Usage:
    from src.agents.question.graph import build_question_graph

    graph = build_question_graph()
    result = await graph.ainvoke(
        {"requirement": {...}, "kb_name": "my_kb", "language": "zh"},
        config={"configurable": {"thread_id": "task_123", "ws_callback": callback}},
    )
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_summary,
    check_has_content,
    check_mode,
    check_more_focuses,
    generate_and_analyze,
    generate_single,
    plan,
    retrieve,
)
from .state import QuestionGraphState


def build_question_graph(
    *,
    enable_checkpoint: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Any:
    """
    Build and compile the question generation graph.

    This graph handles both single and custom (multi-question) modes
    via a conditional edge after retrieval.

    Args:
        enable_checkpoint: Whether to enable SQLite checkpointing.
        checkpoint_path: Path for the checkpoint database.

    Returns:
        Compiled LangGraph graph ready for ainvoke().
    """
    workflow = StateGraph(QuestionGraphState)

    # --- Add Nodes ---
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_single", generate_single)
    workflow.add_node("plan", plan)
    workflow.add_node("generate_and_analyze", generate_and_analyze)
    workflow.add_node("build_summary", build_summary)

    # --- Edges ---

    # START → retrieve
    workflow.add_edge(START, "retrieve")

    # retrieve → check content
    workflow.add_conditional_edges(
        "retrieve",
        check_has_content,
        {
            "has_content": "route_mode",
            "no_content": END,
        },
    )

    # Virtual routing node (implemented via conditional edge from retrieve)
    # We need an intermediate routing step, so let's add a pass-through node
    workflow.add_node("route_mode", _passthrough)
    workflow.add_conditional_edges(
        "route_mode",
        check_mode,
        {
            "single": "generate_single",
            "custom": "plan",
        },
    )

    # Single mode: generate_single → END
    workflow.add_edge("generate_single", END)

    # Custom mode: plan → generate_and_analyze
    workflow.add_edge("plan", "generate_and_analyze")

    # generate_and_analyze → loop or done
    workflow.add_conditional_edges(
        "generate_and_analyze",
        check_more_focuses,
        {
            "more": "generate_and_analyze",
            "done": "build_summary",
        },
    )

    # build_summary → END
    workflow.add_edge("build_summary", END)

    # --- Compile ---
    checkpointer = None
    if enable_checkpoint:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = checkpoint_path or Path("data/checkpoints/question_graph.sqlite")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except Exception:
            checkpointer = None

    return workflow.compile(checkpointer=checkpointer)


async def _passthrough(state: QuestionGraphState, config: RunnableConfig) -> dict:
    """Pass-through node used for routing."""
    return {}


__all__ = ["build_question_graph"]
