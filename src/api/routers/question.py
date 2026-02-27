import asyncio
import base64
from datetime import datetime
from pathlib import Path
import re
import sys
import traceback

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from src.api.dependencies.auth import get_current_user_from_authorization
from src.api.utils.history import ActivityType, history_manager
from src.api.utils.task_id_manager import TaskIDManager
from src.tools.question import mimic_exam_questions
from src.utils.document_validator import DocumentValidator
from src.utils.error_utils import format_exception_message

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logging import get_logger
from src.services.config import load_config_with_main
from src.services.llm.config import get_llm_config
from src.services.settings.interface_settings import get_ui_language
# Setup module logger with unified logging system (from config)
project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("question_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("QuestionAPI", log_dir=log_dir)

router = APIRouter()

# Output directory for mimic mode - use data/user/question
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MIMIC_OUTPUT_DIR = PROJECT_ROOT / "data" / "user" / "question" / "mimic_papers"


def _get_websocket_authorization(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization:
        return authorization

    query_token = websocket.query_params.get("token")
    if query_token:
        return f"Bearer {query_token}"

    return None

@router.websocket("/mimic")
async def websocket_mimic_generate(websocket: WebSocket):
    """
    WebSocket endpoint for mimic exam paper question generation.

    Supports two modes:
    1. Upload PDF directly via WebSocket (base64 encoded)
    2. Use a pre-parsed paper directory path

    Message format for PDF upload:
    {
        "mode": "upload",
        "pdf_data": "base64_encoded_pdf_content",
        "pdf_name": "exam.pdf",
        "kb_name": "knowledge_base_name",
        "max_questions": 5  // optional
    }

    Message format for pre-parsed:
    {
        "mode": "parsed",
        "paper_path": "directory_name",
        "kb_name": "knowledge_base_name",
        "max_questions": 5  // optional
    }
    """
    authorization = _get_websocket_authorization(websocket)
    try:
        get_current_user_from_authorization(authorization)
    except HTTPException as exc:
        await websocket.accept()
        await websocket.send_json({"type": "error", "content": exc.detail})
        await websocket.close(code=1008)
        return

    await websocket.accept()

    pusher_task = None
    original_stdout = sys.stdout

    try:
        # 1. Wait for config
        data = await websocket.receive_json()
        mode = data.get("mode", "parsed")  # "upload" or "parsed"
        kb_name = data.get("kb_name", "ai_textbook")
        max_questions = data.get("max_questions")

        logger.info(f"Starting mimic generation (mode: {mode}, kb: {kb_name})")

        # 2. Setup Log Queue
        log_queue = asyncio.Queue()

        async def log_pusher():
            while True:
                entry = await log_queue.get()
                try:
                    await websocket.send_json(entry)
                except Exception:
                    break
                log_queue.task_done()

        pusher_task = asyncio.create_task(log_pusher())

        # 3. Stdout interceptor for capturing prints
        # ANSI escape sequence pattern for stripping color codes
        ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

        class StdoutInterceptor:
            def __init__(self, queue, original):
                self.queue = queue
                self.original_stdout = original
                self._closed = False

            def write(self, message):
                if self._closed:
                    return
                # Write to terminal first (with ANSI codes for color)
                try:
                    self.original_stdout.write(message)
                except Exception:
                    pass
                # Strip ANSI escape codes before sending to frontend
                clean_message = ANSI_ESCAPE_PATTERN.sub("", message).strip()
                # Then send to frontend (non-blocking)
                if clean_message:
                    try:
                        self.queue.put_nowait(
                            {
                                "type": "log",
                                "content": clean_message,
                                "timestamp": asyncio.get_event_loop().time(),
                            }
                        )
                    except (asyncio.QueueFull, RuntimeError):
                        pass

            def flush(self):
                if not self._closed:
                    try:
                        self.original_stdout.flush()
                    except Exception:
                        pass

            def close(self):
                """Mark interceptor as closed to prevent further writes."""
                self._closed = True

        interceptor = StdoutInterceptor(log_queue, original_stdout)
        sys.stdout = interceptor

        try:
            await websocket.send_json(
                {"type": "status", "stage": "init", "content": "Initializing..."}
            )

            pdf_path = None
            paper_dir = None

            # Handle PDF upload mode
            if mode == "upload":
                pdf_data = data.get("pdf_data")
                pdf_name = data.get("pdf_name", "exam.pdf")

                if not pdf_data:
                    await websocket.send_json(
                        {"type": "error", "content": "PDF data is required for upload mode"}
                    )
                    return

                # Decode PDF data first to check size
                try:
                    pdf_bytes = base64.b64decode(pdf_data)
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "content": f"Invalid base64 PDF data: {e}"}
                    )
                    return

                # Pre-validate filename and file size before writing
                try:
                    safe_name = DocumentValidator.validate_upload_safety(
                        pdf_name, len(pdf_bytes), {".pdf"}
                    )
                except ValueError as e:
                    await websocket.send_json({"type": "error", "content": str(e)})
                    return

                # Create batch directory for this mimic session
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_stem = Path(safe_name).stem
                batch_dir = MIMIC_OUTPUT_DIR / f"mimic_{timestamp}_{pdf_stem}"
                batch_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded PDF in batch directory
                pdf_path = batch_dir / safe_name

                await websocket.send_json(
                    {"type": "status", "stage": "upload", "content": f"Saving PDF: {safe_name}"}
                )

                # Write the validated PDF bytes
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                # Additional validation (file readability, etc.)
                try:
                    DocumentValidator.validate_file(pdf_path)
                except (ValueError, FileNotFoundError, PermissionError) as e:
                    # Clean up invalid or inaccessible file
                    pdf_path.unlink(missing_ok=True)
                    await websocket.send_json({"type": "error", "content": str(e)})
                    return

                await websocket.send_json(
                    {
                        "type": "status",
                        "stage": "parsing",
                        "content": "Parsing PDF exam paper (MinerU)...",
                    }
                )
                logger.info(f"Saved and validated uploaded PDF to: {pdf_path}")

                # Pass batch_dir as output directory
                pdf_path = str(pdf_path)
                output_dir = str(batch_dir)

            elif mode == "parsed":
                paper_path = data.get("paper_path")
                if not paper_path:
                    await websocket.send_json(
                        {"type": "error", "content": "paper_path is required for parsed mode"}
                    )
                    return
                paper_dir = paper_path

                # Create batch directory for parsed mode too
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = MIMIC_OUTPUT_DIR / f"mimic_{timestamp}_{Path(paper_path).name}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(batch_dir)

            else:
                await websocket.send_json({"type": "error", "content": f"Unknown mode: {mode}"})
                return

            # Create WebSocket callback for real-time progress updates
            async def ws_callback(event_type: str, data: dict):
                """Send progress updates to the frontend via WebSocket."""
                try:
                    message = {"type": event_type, **data}
                    await websocket.send_json(message)
                except Exception as e:
                    logger.debug(f"WebSocket send failed: {e}")

            # Run the complete mimic workflow with callback
            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "processing",
                    "content": "Executing question generation workflow...",
                }
            )

            result = await mimic_exam_questions(
                pdf_path=pdf_path,
                paper_dir=paper_dir,
                kb_name=kb_name,
                output_dir=output_dir,
                max_questions=max_questions,
                ws_callback=ws_callback,
            )

            if result.get("success"):
                # Results are already sent via ws_callback during generation
                # Just send the final complete signal
                total_ref = result.get("total_reference_questions", 0)
                generated = result.get("generated_questions", [])
                failed = result.get("failed_questions", [])

                logger.success(
                    f"Mimic generation complete: {len(generated)} succeeded, {len(failed)} failed"
                )

                try:
                    await websocket.send_json({"type": "complete"})
                except (RuntimeError, WebSocketDisconnect):
                    logger.debug("WebSocket closed before complete signal could be sent")
            else:
                error_msg = result.get("error", "Unknown error")
                try:
                    await websocket.send_json({"type": "error", "content": error_msg})
                except (RuntimeError, WebSocketDisconnect):
                    pass
                logger.error(f"Mimic generation failed: {error_msg}")

        finally:
            # Close interceptor and restore stdout
            if "interceptor" in locals():
                interceptor.close()
            sys.stdout = original_stdout

    except WebSocketDisconnect:
        logger.debug("Client disconnected during mimic generation")
    except Exception as e:
        logger.exception("Mimic generation error")
        error_msg = format_exception_message(e)
        try:
            await websocket.send_json({"type": "error", "content": error_msg})
        except Exception:
            pass
    finally:
        # Ensure stdout is always restored
        sys.stdout = original_stdout

        # Clean up pusher task
        if pusher_task:
            try:
                pusher_task.cancel()
                await pusher_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except Exception:
                pass

        # Drain any remaining items in the queue
        try:
            while not log_queue.empty():
                log_queue.get_nowait()
        except Exception:
            pass

        # Close WebSocket
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/generate")
async def websocket_question_generate(websocket: WebSocket):
    authorization = _get_websocket_authorization(websocket)
    try:
        current_user = get_current_user_from_authorization(authorization)
    except HTTPException as exc:
        await websocket.accept()
        await websocket.send_json({"type": "error", "content": exc.detail})
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # Get task ID manager
    task_manager = TaskIDManager.get_instance()

    try:
        # 1. Wait for config
        data = await websocket.receive_json()
        requirement = data.get("requirement")
        kb_name = data.get("kb_name", "ai_textbook")
        count = data.get("count", 1)

        if not requirement:
            try:
                await websocket.send_json({"type": "error", "content": "Requirement is required"})
            except (RuntimeError, WebSocketDisconnect):
                pass
            return

        # Generate task ID
        task_key = f"question_{kb_name}_{hash(str(requirement))}"
        task_id = task_manager.generate_task_id("question_gen", task_key)

        # Send task ID to frontend
        try:
            await websocket.send_json({"type": "task_id", "task_id": task_id})
        except (RuntimeError, WebSocketDisconnect):
            logger.debug("WebSocket closed, cannot send task_id")
            return

        logger.info(
            f"[{task_id}] Starting question generation: {requirement.get('knowledge_point', 'Unknown')}"
        )

        language = get_ui_language(
            user_id=current_user["id"],
            default=config.get("system", {}).get("language", "en"),
        )

        # Define unified output directory (DeepTutor/data/user/question)
        root_dir = Path(__file__).parent.parent.parent.parent
        output_base = root_dir / "data" / "user" / "question"

        # 3. Setup Log Queue for WebSocket streaming
        log_queue = asyncio.Queue()

        # WebSocket callback for sending structured updates
        async def ws_callback(data: dict):
            try:
                await log_queue.put(data)
            except Exception:
                pass

        # 4. Define background pusher for logs
        async def log_pusher():
            while True:
                entry = await log_queue.get()
                try:
                    await websocket.send_json(entry)
                except Exception:
                    break
                log_queue.task_done()

        pusher_task = asyncio.create_task(log_pusher())

        try:
            try:
                await websocket.send_json({"type": "status", "content": "started"})
            except (RuntimeError, WebSocketDisconnect):
                logger.debug("WebSocket closed, stopping question generation")
                return

            # Use LangGraph generation
            await _run_langgraph_generation(
                websocket=websocket,
                log_queue=log_queue,
                requirement=requirement,
                kb_name=kb_name,
                count=count,
                language=language,
                output_dir=str(output_base),
                task_id=task_id,
                task_manager=task_manager,
                user_id=current_user["id"],
            )

        except Exception as e:
            error_msg = format_exception_message(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Question generation error: {error_msg}")
            logger.error(f"Error traceback:\n{error_traceback}")

            try:
                await websocket.send_json({"type": "error", "content": error_msg})
            except (RuntimeError, WebSocketDisconnect):
                logger.debug("WebSocket closed, cannot send error message")
            task_manager.update_task_status(task_id, "error", error=error_msg)

        finally:
            pusher_task.cancel()
            try:
                await pusher_task
            except asyncio.CancelledError:
                pass
            await websocket.close()

    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as e:
        error_msg = format_exception_message(e)
        logger.error(f"WebSocket error: {error_msg}")


# ---------------------------------------------------------------------------
# Helper: LangGraph generation
# ---------------------------------------------------------------------------

async def _run_langgraph_generation(
    *,
    websocket: WebSocket,
    log_queue: asyncio.Queue,
    requirement: dict,
    kb_name: str,
    count: int,
    language: str,
    output_dir: str,
    task_id: str,
    task_manager: TaskIDManager,
    user_id: str,
) -> None:
    """Run question generation using the LangGraph StateGraph."""
    from src.agents.question.graph import build_question_graph

    logger.info(f"Starting LangGraph generation for {count} question(s)")

    # WebSocket callback that sends updates through the log queue
    async def ws_callback(data: dict):
        try:
            await log_queue.put(data)
        except Exception:
            pass

    graph = build_question_graph()

    # Build initial state
    initial_state = {
        "requirement": requirement,
        "kb_name": kb_name,
        "language": language,
        "num_questions": count,
        "output_dir": output_dir,
    }

    # Run the graph
    result = await graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": task_id, "ws_callback": ws_callback},
                "callbacks": []},
    )

    # Extract summary from graph result
    summary = result.get("summary", {})
    results = result.get("results", [])

    # Save successful results to history
    for r in results:
        question = r.get("question", {})
        validation = r.get("validation", {})
        history_manager.add_entry(
            user_id=user_id,
            activity_type=ActivityType.QUESTION,
            title=f"{requirement.get('knowledge_point', 'Question')} ({requirement.get('question_type')})",
            content={
                "requirement": requirement,
                "question": question,
                "validation": validation,
                "kb_name": kb_name,
            },
            summary=question.get("question", "")[:100] if isinstance(question, dict) else "",
        )

    # Send token stats if available
    token_stats = result.get("token_stats", {})
    if token_stats:
        try:
            await websocket.send_json({"type": "token_stats", "stats": token_stats})
        except (RuntimeError, WebSocketDisconnect):
            pass

    # Send batch summary
    try:
        await websocket.send_json(
            {
                "type": "batch_summary",
                "requested": summary.get("requested", count),
                "completed": summary.get("completed", len(results)),
                "failed": summary.get("failed", 0),
                "plan": summary.get("plan", {}),
            }
        )
    except (RuntimeError, WebSocketDisconnect):
        pass

    if not summary.get("success", len(results) > 0):
        logger.warning(
            f"LangGraph generation had failures: {summary.get('failed', 0)} failed"
        )

    # Wait for pending messages
    await asyncio.sleep(0.1)
    while not log_queue.empty():
        await asyncio.sleep(0.05)

    # Send complete signal
    try:
        await websocket.send_json({"type": "complete"})
        logger.info(f"[{task_id}] Question generation completed (langgraph)")
        task_manager.update_task_status(task_id, "completed")
    except (RuntimeError, WebSocketDisconnect):
        logger.debug("WebSocket closed, cannot send complete signal")
