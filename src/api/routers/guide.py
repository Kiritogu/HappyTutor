"""
Guided Learning API Router
==========================

Provides session creation, learning progress management, and chat interaction.
"""

from pathlib import Path
import sys

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.base_agent import BaseAgent
from src.api.dependencies.auth import (
    get_current_user_from_authorization,
    get_current_user_from_header,
)
from src.api.utils.notebook_manager import notebook_manager
from src.api.utils.task_id_manager import TaskIDManager
from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.settings.interface_settings import get_ui_language

router = APIRouter()

# Initialize logger with config
project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("guide_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("Guide", level="INFO", log_dir=log_dir)


def _get_websocket_authorization(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization:
        return authorization

    query_token = websocket.query_params.get("token")
    if query_token:
        return f"Bearer {query_token}"

    return None


# === Request/Response Models ===


class CreateSessionRequest(BaseModel):
    """Create session request"""

    notebook_id: str | None = None  # Optional, single notebook mode
    records: list[dict] | None = None  # Optional, cross-notebook mode with direct records


class ChatRequest(BaseModel):
    """Chat request"""

    session_id: str
    message: str


class FixHtmlRequest(BaseModel):
    """Fix HTML request"""

    session_id: str
    bug_description: str


class NextKnowledgeRequest(BaseModel):
    """Next knowledge point request"""

    session_id: str


# === REST API Endpoints ===


@router.post("/create_session")
async def create_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Create a new guided learning session.

    Returns:
        Session creation result with knowledge point list.
    """
    task_manager = TaskIDManager.get_instance()

    try:
        records = []
        notebook_name = "Unknown"

        # Mode 1: Cross-notebook mode - use provided records directly
        if request.records and isinstance(request.records, list):
            records = request.records
            notebook_name = f"Cross-notebook ({len(records)} records)"
        # Mode 2: Single notebook mode - get records from notebook
        elif request.notebook_id:
            notebook = notebook_manager.get_notebook(
                user_id=current_user["id"],
                notebook_id=request.notebook_id,
            )
            if not notebook:
                raise HTTPException(status_code=404, detail="Notebook not found")

            records = notebook.get("records", [])
            notebook_name = notebook.get("name", "Unknown")
        else:
            raise HTTPException(status_code=400, detail="Must provide notebook_id or records")

        if not records:
            raise HTTPException(status_code=400, detail="No available records")

        # Reset LLM stats for new session
        BaseAgent.reset_stats("guide")

        ui_language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        # Use LangGraph for session creation
        result = await _create_session_langgraph(
            user_id=current_user["id"],
            notebook_id=request.notebook_id or "cross_notebook",
            notebook_name=notebook_name,
            records=records,
            language=ui_language,
        )

        if result and "session_id" in result:
            session_id = result["session_id"]
            task_id = task_manager.generate_task_id("guide", session_id)
            logger.info(f"[{task_id}] Session created: {session_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_learning(
    request: NextKnowledgeRequest,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Start learning (get the first knowledge point).
    """
    try:
        ui_language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        result = await _run_interaction_langgraph(
            user_id=current_user["id"],
            session_id=request.session_id,
            action="start",
            language=ui_language,
        )
        return result
    except Exception as e:
        logger.error(f"Start learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/next")
async def next_knowledge(
    request: NextKnowledgeRequest,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Move to the next knowledge point.
    """
    try:
        ui_language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        result = await _run_interaction_langgraph(
            user_id=current_user["id"],
            session_id=request.session_id,
            action="next",
            language=ui_language,
        )

        # Print stats if learning completed
        if result.get("learning_complete", False):
            BaseAgent.print_stats("guide")

        return result
    except Exception as e:
        logger.error(f"Next knowledge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Send a chat message.
    """
    try:
        ui_language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        result = await _run_interaction_langgraph(
            user_id=current_user["id"],
            session_id=request.session_id,
            action="chat",
            language=ui_language,
            user_message=request.message,
        )
        return result
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix_html")
async def fix_html(
    request: FixHtmlRequest,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Fix HTML page bugs.
    """
    try:
        ui_language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        result = await _run_interaction_langgraph(
            user_id=current_user["id"],
            session_id=request.session_id,
            action="fix_html",
            language=ui_language,
            bug_description=request.bug_description,
        )
        return result
    except Exception as e:
        logger.error(f"Fix HTML failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Get session information.
    """
    try:
        session = _langgraph_sessions.get(session_id)
        if not session or session.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/html")
async def get_current_html(
    session_id: str,
    current_user: dict = Depends(get_current_user_from_header),
):
    """
    Get the current HTML page.
    """
    try:
        session = _langgraph_sessions.get(session_id)
        if not session or session.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=404, detail="Session not found")
        html = session.get("current_html")
        if html is None:
            raise HTTPException(status_code=404, detail="No HTML content")
        return {"html": html}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get HTML failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === WebSocket Endpoint ===


@router.websocket("/ws/{session_id}")
async def websocket_guide(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time interaction.

    Message types:
    - start: Start learning
    - next: Next knowledge point
    - chat: Send chat message
    - fix_html: Fix HTML
    - get_session: Get session state
    """
    authorization = _get_websocket_authorization(websocket)
    try:
        current_user = get_current_user_from_authorization(authorization)
    except HTTPException as exc:
        await websocket.accept()
        await websocket.send_json({"type": "error", "content": exc.detail})
        await websocket.close(code=1008)
        return

    await websocket.accept()

    task_manager = TaskIDManager.get_instance()
    task_id = task_manager.generate_task_id("guide", session_id)

    try:
        await websocket.send_json({"type": "task_id", "task_id": task_id})
    except (RuntimeError, WebSocketDisconnect, ConnectionError) as e:
        logger.debug(f"Failed to send task_id: {e}")

    try:
        # Get session from LangGraph session store
        session = _langgraph_sessions.get(session_id)
        if not session or session.get("user_id") != current_user["id"]:
            await websocket.send_json({
                "type": "error",
                "content": "Session not found or expired. Please create a new session.",
                "session_expired": True,
            })
            await websocket.close()
            return

        logger.info(f"[{task_id}] Guide session started: {session_id}")

        await websocket.send_json({"type": "session_info", "data": session})

        ui_language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        # Create WebSocket callback for real-time updates from LangGraph nodes
        async def ws_callback(data: dict):
            try:
                await websocket.send_json(data)
            except Exception:
                pass

        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "start":
                    logger.debug(f"[{task_id}] Start learning")
                    result = await _run_interaction_langgraph(
                        user_id=current_user["id"],
                        session_id=session_id,
                        action="start",
                        language=ui_language,
                        ws_callback=ws_callback,
                    )
                    await websocket.send_json({"type": "start_result", "data": result})

                elif msg_type == "next":
                    logger.debug(f"[{task_id}] Next knowledge point")
                    result = await _run_interaction_langgraph(
                        user_id=current_user["id"],
                        session_id=session_id,
                        action="next",
                        language=ui_language,
                        ws_callback=ws_callback,
                    )
                    await websocket.send_json({"type": "next_result", "data": result})

                    # Print stats if learning completed
                    if result.get("learning_complete", False):
                        BaseAgent.print_stats("guide")

                elif msg_type == "chat":
                    message = data.get("message", "")
                    if message:
                        logger.debug(f"[{task_id}] User message: {message[:50]}...")
                        result = await _run_interaction_langgraph(
                            user_id=current_user["id"],
                            session_id=session_id,
                            action="chat",
                            language=ui_language,
                            user_message=message,
                            ws_callback=ws_callback,
                        )
                        await websocket.send_json({"type": "chat_result", "data": result})

                elif msg_type == "fix_html":
                    bug_desc = data.get("bug_description", "")
                    logger.debug(f"[{task_id}] Fix HTML: {bug_desc[:50]}...")
                    result = await _run_interaction_langgraph(
                        user_id=current_user["id"],
                        session_id=session_id,
                        action="fix_html",
                        language=ui_language,
                        bug_description=bug_desc,
                        ws_callback=ws_callback,
                    )
                    await websocket.send_json({"type": "fix_result", "data": result})

                elif msg_type == "get_session":
                    session = _langgraph_sessions.get(session_id)
                    if session and session.get("user_id") == current_user["id"]:
                        await websocket.send_json({"type": "session_info", "data": session})
                    else:
                        await websocket.send_json({"type": "error", "content": "Session not found"})

                else:
                    await websocket.send_json(
                        {"type": "error", "content": f"Unknown message type: {msg_type}"}
                    )

            except WebSocketDisconnect:
                logger.debug(f"WebSocket disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({"type": "error", "content": str(e)})

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect, ConnectionError):
            pass  # Connection already closed


@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "guide"}


# ---------------------------------------------------------------------------
# LangGraph Helper Functions
# ---------------------------------------------------------------------------

# In-memory session store for LangGraph mode (shared state between requests)
_langgraph_sessions: dict[str, dict] = {}


async def _create_session_langgraph(
    user_id: str,
    notebook_id: str,
    notebook_name: str,
    records: list[dict],
    language: str,
) -> dict:
    """Create a guided learning session using LangGraph."""
    from src.agents.guide.graph import build_session_graph

    logger.info(f"Creating session via LangGraph for notebook: {notebook_name}")

    graph = build_session_graph()

    initial_state = {
        "notebook_id": notebook_id,
        "notebook_name": notebook_name,
        "records": records,
    }

    result = await graph.ainvoke(
        initial_state,
        config={"configurable": {"language": language}},
    )

    if result.get("success"):
        session_id = result.get("session_id")
        # Store session state for later interactions
        _langgraph_sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "notebook_id": notebook_id,
            "notebook_name": notebook_name,
            "knowledge_points": result.get("knowledge_points", []),
            "current_index": 0,
            "status": "initialized",
            "chat_history": [],
            "current_html": "",
            "summary": "",
        }

    return {
        "success": result.get("success", False),
        "session_id": result.get("session_id"),
        "knowledge_points": result.get("knowledge_points", []),
        "total_points": result.get("total_points", 0),
        "error": result.get("error"),
    }


async def _run_interaction_langgraph(
    user_id: str,
    session_id: str,
    action: str,
    language: str,
    user_message: str = "",
    bug_description: str = "",
    ws_callback=None,
) -> dict:
    """Run a user interaction using LangGraph."""
    from src.agents.guide.graph import build_interaction_graph

    # Get session from memory store
    session = _langgraph_sessions.get(session_id)
    if not session or session.get("user_id") != user_id:
        # Session not found (possibly server restarted)
        # Return special error so frontend knows to reset
        return {
            "success": False,
            "error": "Session not found or expired. Please create a new session.",
            "session_expired": True,  # Frontend should clear local state and restart
        }

    logger.info(f"Running LangGraph interaction: action={action}, session={session_id}")

    graph = build_interaction_graph()

    # Build state from stored session + current action
    state = {
        "session_id": session_id,
        "notebook_id": session.get("notebook_id", ""),
        "notebook_name": session.get("notebook_name", ""),
        "knowledge_points": session.get("knowledge_points", []),
        "current_index": session.get("current_index", 0),
        "status": session.get("status", "initialized"),
        "chat_history": session.get("chat_history", []),
        "current_html": session.get("current_html", ""),
        "action": action,
        "user_message": user_message,
        "bug_description": bug_description,
    }

    # Build config with optional ws_callback
    run_config = {
        "configurable": {"thread_id": session_id, "language": language},
        "callbacks": [],
    }
    if ws_callback:
        run_config["configurable"]["ws_callback"] = ws_callback

    result = await graph.ainvoke(state, config=run_config)

    # Update session store with new state
    _langgraph_sessions[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "notebook_id": result.get("notebook_id", session.get("notebook_id")),
        "notebook_name": result.get("notebook_name", session.get("notebook_name")),
        "knowledge_points": result.get("knowledge_points", session.get("knowledge_points")),
        "current_index": result.get("current_index", session.get("current_index")),
        "status": result.get("status", session.get("status")),
        "chat_history": result.get("chat_history", session.get("chat_history")),
        "current_html": result.get("current_html", session.get("current_html")),
        "summary": result.get("summary", session.get("summary", "")),
    }

    # Return the result dict from the graph
    return result.get("result", {"success": False, "error": "No result from graph"})
