# -*- coding: utf-8 -*-
"""
Guide Graph Nodes
==================

Node functions for LangGraph guided learning workflows.
Each node wraps an existing agent's process() method and returns state updates.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from .state import CreateSessionState, GuideGraphState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ws_callback(config: RunnableConfig, update_type: str, data: dict[str, Any]) -> None:
    """Send a WebSocket update if a callback is configured."""
    cb = config.get("configurable", {}).get("ws_callback")
    if cb:
        try:
            await cb({"type": update_type, **data})
        except Exception:
            pass


def _get_current_knowledge(state: GuideGraphState) -> dict[str, Any] | None:
    """Get the current knowledge point from state."""
    knowledge_points = state.get("knowledge_points", [])
    current_index = state.get("current_index", 0)
    if 0 <= current_index < len(knowledge_points):
        return knowledge_points[current_index]
    return None


def _compute_progress(state: GuideGraphState) -> int:
    """Compute learning progress percentage."""
    knowledge_points = state.get("knowledge_points", [])
    current_index = state.get("current_index", 0)
    total = len(knowledge_points)
    if total == 0:
        return 0
    return round((current_index + 1) / total * 100)


# ---------------------------------------------------------------------------
# Session Creation Graph Nodes
# ---------------------------------------------------------------------------

async def locate_knowledge(state: CreateSessionState, config: RunnableConfig) -> dict:
    """Analyze notebook content and identify knowledge points using LocateAgent."""
    from src.agents.guide.agents.locate_agent import LocateAgent

    await _ws_callback(config, "progress", {"stage": "locating", "status": "analyzing"})

    records = state.get("records", [])
    if not records:
        return {
            "success": False,
            "error": "No records provided",
            "knowledge_points": [],
            "total_points": 0,
        }

    language = config.get("configurable", {}).get("language", "en")
    agent = LocateAgent(language=language)

    result = await agent.process(
        notebook_id=state.get("notebook_id", ""),
        notebook_name=state.get("notebook_name", "Unknown"),
        records=records,
    )

    knowledge_points = result.get("knowledge_points", [])

    await _ws_callback(config, "knowledge_located", {
        "total_points": len(knowledge_points),
        "knowledge_points": knowledge_points,
    })

    # Generate session ID
    import time
    session_id = f"guide_{int(time.time() * 1000)}"

    return {
        "session_id": session_id,
        "knowledge_points": knowledge_points,
        "total_points": len(knowledge_points),
        "success": result.get("success", False),
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------------
# Interaction Graph Nodes
# ---------------------------------------------------------------------------

async def generate_interactive(state: GuideGraphState, config: RunnableConfig) -> dict:
    """Generate interactive HTML page for current knowledge point."""
    from src.agents.guide.agents.interactive_agent import InteractiveAgent

    knowledge = _get_current_knowledge(state)
    if not knowledge:
        return {
            "error": "No current knowledge point",
            "result": {"success": False, "error": "No current knowledge point"},
        }

    await _ws_callback(config, "progress", {
        "stage": "generating",
        "knowledge_title": knowledge.get("knowledge_title", ""),
    })

    language = config.get("configurable", {}).get("language", "en")
    agent = InteractiveAgent(language=language)

    result = await agent.process(knowledge=knowledge)

    html = result.get("html", "")
    progress = _compute_progress(state)
    current_index = state.get("current_index", 0)
    total_points = len(state.get("knowledge_points", []))

    await _ws_callback(config, "html_ready", {
        "html": html,
        "knowledge": knowledge,
        "progress": progress,
    })

    return {
        "current_html": html,
        "current_knowledge": knowledge,
        "progress": progress,
        "status": "learning",
        "result": {
            "success": True,
            "current_index": current_index,
            "current_knowledge": knowledge,
            "html": html,
            "progress": progress,
            "total_points": total_points,
            "message": f"Learning knowledge point {current_index + 1}/{total_points}",
            "is_fallback": result.get("is_fallback", False),
        },
    }


async def handle_chat(state: GuideGraphState, config: RunnableConfig) -> dict:
    """Handle user chat message using ChatAgent."""
    from src.agents.guide.agents.chat_agent import ChatAgent

    knowledge = _get_current_knowledge(state)
    if not knowledge:
        return {
            "assistant_response": "No current knowledge point to discuss.",
            "result": {"success": False, "error": "No current knowledge point"},
        }

    user_message = state.get("user_message", "")
    if not user_message.strip():
        return {
            "assistant_response": "Please provide a question.",
            "result": {"success": False, "error": "Empty message"},
        }

    await _ws_callback(config, "progress", {"stage": "chatting", "status": "processing"})

    language = config.get("configurable", {}).get("language", "en")
    agent = ChatAgent(language=language)

    chat_history = list(state.get("chat_history", []))

    result = await agent.process(
        knowledge=knowledge,
        chat_history=chat_history,
        user_question=user_message,
    )

    answer = result.get("answer", "Sorry, I couldn't process your question.")

    # Update chat history
    current_index = state.get("current_index", 0)
    chat_history.append({
        "role": "user",
        "content": user_message,
        "knowledge_index": current_index,
    })
    chat_history.append({
        "role": "assistant",
        "content": answer,
        "knowledge_index": current_index,
    })

    await _ws_callback(config, "chat_response", {
        "answer": answer,
        "knowledge_index": current_index,
    })

    return {
        "assistant_response": answer,
        "chat_history": chat_history,
        "result": {
            "success": True,
            "answer": answer,
            "knowledge_index": current_index,
        },
    }


async def fix_interactive(state: GuideGraphState, config: RunnableConfig) -> dict:
    """Fix HTML page bugs using InteractiveAgent."""
    from src.agents.guide.agents.interactive_agent import InteractiveAgent

    knowledge = _get_current_knowledge(state)
    if not knowledge:
        return {
            "result": {"success": False, "error": "No current knowledge point"},
        }

    bug_description = state.get("bug_description", "")
    if not bug_description.strip():
        return {
            "result": {"success": False, "error": "No bug description provided"},
        }

    await _ws_callback(config, "progress", {"stage": "fixing", "status": "regenerating"})

    language = config.get("configurable", {}).get("language", "en")
    agent = InteractiveAgent(language=language)

    result = await agent.process(knowledge=knowledge, retry_with_bug=bug_description)

    html = result.get("html", "")

    await _ws_callback(config, "html_fixed", {
        "html": html,
        "is_fallback": result.get("is_fallback", False),
    })

    return {
        "current_html": html,
        "result": {
            "success": True,
            "html": html,
            "is_fallback": result.get("is_fallback", False),
        },
    }


async def advance_index(state: GuideGraphState, config: RunnableConfig) -> dict:
    """Advance to the next knowledge point index."""
    current_index = state.get("current_index", 0)
    new_index = current_index + 1
    knowledge_points = state.get("knowledge_points", [])
    learning_complete = new_index >= len(knowledge_points)

    return {
        "current_index": new_index,
        "learning_complete": learning_complete,
    }


async def generate_summary(state: GuideGraphState, config: RunnableConfig) -> dict:
    """Generate learning summary using SummaryAgent."""
    from src.agents.guide.agents.summary_agent import SummaryAgent

    await _ws_callback(config, "progress", {"stage": "summarizing", "status": "generating"})

    language = config.get("configurable", {}).get("language", "en")
    agent = SummaryAgent(language=language)

    result = await agent.process(
        notebook_name=state.get("notebook_name", "Unknown"),
        knowledge_points=state.get("knowledge_points", []),
        chat_history=state.get("chat_history", []),
    )

    summary = result.get("summary", "Learning completed!")

    await _ws_callback(config, "learning_complete", {
        "summary": summary,
        "total_points": result.get("total_points", 0),
        "total_interactions": result.get("total_interactions", 0),
    })

    return {
        "summary": summary,
        "status": "completed",
        "result": {
            "success": True,
            "status": "completed",  # Frontend expects this field
            "learning_complete": True,
            "summary": summary,
            "message": "Congratulations! You have completed all knowledge points.",
        },
    }


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_action(state: GuideGraphState) -> str:
    """Route based on the action type."""
    action = state.get("action", "")
    if action == "start":
        return "start"
    elif action == "next":
        return "next"
    elif action == "chat":
        return "chat"
    elif action == "fix_html":
        return "fix_html"
    return "invalid"


def check_learning_complete(state: GuideGraphState) -> str:
    """Check if learning is complete after advancing index."""
    if state.get("learning_complete", False):
        return "complete"
    return "continue"


__all__ = [
    "locate_knowledge",
    "generate_interactive",
    "handle_chat",
    "fix_interactive",
    "advance_index",
    "generate_summary",
    "route_action",
    "check_learning_complete",
]
