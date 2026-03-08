#!/usr/bin/env python
"""
SessionManager - Chat session persistence and management.

This module handles:
- Creating new chat sessions
- Updating sessions with new messages
- Retrieving session history
- Listing recent sessions
- Deleting sessions
"""

from pathlib import Path
from typing import Any

from src.services.storage import get_user_db


class SessionManager:
    """
    Manages persistent storage of chat sessions via PostgreSQL.

    Each session contains:
    - session_id: Unique identifier
    - title: Session title (usually first user message)
    - messages: List of messages with role, content, sources, timestamp
    - settings: RAG/Web Search settings used
    - created_at: Creation timestamp
    - updated_at: Last update timestamp
    """

    def __init__(self, base_dir: str | None = None):
        """
        Initialize SessionManager.

        Args:
            base_dir: Base directory for output files.
                     Defaults to project_root/data/user
        """
        project_root = Path(__file__).resolve().parents[3]

        if base_dir is None:
            # Current file: src/agents/chat/session_manager.py
            # Project root: 4 levels up
            base_dir_path = project_root / "data" / "user"
        else:
            base_dir_path = Path(base_dir)

        self.base_dir = base_dir_path
        self.base_dir.mkdir(parents=True, exist_ok=True)

        from src.services.storage import get_user_db

        self._db = get_user_db(project_root=project_root)

    def create_session(
        self,
        *,
        user_id: str,
        title: str = "New Chat",
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new chat session.

        Args:
            title: Session title
            settings: Optional settings (kb_name, enable_rag, enable_web_search)

        Returns:
            New session dict with session_id
        """
        return self._db.chat_create_session(user_id=user_id, title=title, settings=settings)

    def get_session(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session dict or None if not found
        """
        return self._db.chat_get_session(user_id=user_id, session_id=session_id)

    def update_session(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Update a session with new data.

        Args:
            session_id: Session identifier
            messages: New messages list (replaces existing)
            title: New title (optional)
            settings: New settings (optional)

        Returns:
            Updated session or None if not found
        """
        return self._db.chat_update_session(
            session_id,
            user_id=user_id,
            messages=messages,
            title=title,
            settings=settings,
        )

    def add_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        sources: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Add a single message to a session.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            sources: Optional sources dict (for assistant messages)

        Returns:
            Updated session or None if not found
        """
        return self._db.chat_add_message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
        )

    def list_sessions(
        self,
        *,
        user_id: str,
        limit: int = 20,
        include_messages: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return
            include_messages: Whether to include full message history

        Returns:
            List of session dicts (newest first)
        """
        return self._db.chat_list_sessions(
            user_id=user_id,
            limit=limit,
            include_messages=include_messages,
        )

    def delete_session(self, *, user_id: str, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        return self._db.chat_delete_session(user_id=user_id, session_id=session_id)

    def clear_all_sessions(self, *, user_id: str) -> int:
        """
        Delete all sessions.

        Returns:
            Number of sessions deleted
        """
        return self._db.chat_clear_all_sessions(user_id=user_id)


# Singleton instance for convenience
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the global SessionManager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


__all__ = ["SessionManager", "get_session_manager"]
