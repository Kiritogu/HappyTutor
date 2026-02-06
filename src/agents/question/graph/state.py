# -*- coding: utf-8 -*-
"""
Question Graph State Definitions
=================================

TypedDict state schemas for LangGraph question generation workflows.
"""

from typing import Any, TypedDict


class QuestionGraphState(TypedDict, total=False):
    """State for the question generation graph (both single and custom modes)."""

    # --- Inputs ---
    requirement: dict[str, Any]
    kb_name: str
    language: str
    num_questions: int
    output_dir: str
    rag_query_count: int

    # --- Stage 1: Retrieval ---
    knowledge_context: str
    has_content: bool
    queries: list[str]
    retrieval_result: dict[str, Any]

    # --- Stage 2: Planning (custom mode only) ---
    plan: dict[str, Any]
    focuses: list[dict[str, Any]]

    # --- Stage 3: Generation loop ---
    current_focus_index: int
    results: list[dict[str, Any]]
    failures: list[dict[str, Any]]

    # --- Output ---
    summary: dict[str, Any]
    error: str | None


__all__ = ["QuestionGraphState"]
