# -*- coding: utf-8 -*-
"""
Guide Graph State Definitions
==============================

TypedDict state schemas for LangGraph guided learning workflows.
"""

from typing import Any, TypedDict


class GuideGraphState(TypedDict, total=False):
    """State for the guided learning interaction graph."""

    # --- Session Identity ---
    session_id: str
    notebook_id: str
    notebook_name: str

    # --- Knowledge Points ---
    knowledge_points: list[dict[str, Any]]
    current_index: int
    status: str  # "initialized" | "learning" | "completed"

    # --- Current Interaction ---
    current_html: str
    action: str  # "start" | "next" | "chat" | "fix_html"
    user_message: str
    bug_description: str

    # --- Chat ---
    chat_history: list[dict[str, str]]
    assistant_response: str

    # --- Output ---
    summary: str
    result: dict[str, Any]
    error: str | None

    # --- Progress (computed) ---
    current_knowledge: dict[str, Any] | None
    progress: dict[str, Any]
    learning_complete: bool


class CreateSessionState(TypedDict, total=False):
    """State for the session creation graph (one-time use)."""

    # --- Inputs ---
    notebook_id: str
    notebook_name: str
    records: list[dict[str, Any]]

    # --- Output ---
    session_id: str
    knowledge_points: list[dict[str, Any]]
    total_points: int
    success: bool
    error: str | None


__all__ = ["GuideGraphState", "CreateSessionState"]
