# -*- coding: utf-8 -*-
"""
Research Graph Nodes
=====================

Node functions for LangGraph research workflows.
Each node wraps an existing agent's process() method and returns state updates.

The graph implements a three-phase research pipeline:
1. Planning: rephrase → decompose → initialize_queue
2. Researching: select_next_block → research_block → mark_complete (loop)
3. Reporting: generate_report → save_results
"""

from __future__ import annotations

from datetime import datetime
import inspect
import json
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from .state import ResearchGraphState

# Registry to store pipeline instances by research_id (for backward compatibility reference)
_pipeline_registry: dict[str, Any] = {}


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


def _cleanup_registry(research_id: str) -> None:
    """Remove pipeline from registry after completion."""
    _pipeline_registry.pop(research_id, None)


def _normalize_subtopic_payload(subtopic: Any, index: int) -> tuple[str, str]:
    if isinstance(subtopic, dict):
        title = str(
            subtopic.get("sub_topic")
            or subtopic.get("subtopic")
            or subtopic.get("title")
            or subtopic.get("topic")
            or subtopic.get("name")
            or ""
        ).strip()
        overview = str(
            subtopic.get("overview")
            or subtopic.get("description")
            or subtopic.get("summary")
            or ""
        ).strip()
        return title or f"Subtopic {index}", overview

    if isinstance(subtopic, str):
        title = subtopic.strip()
        return title or f"Subtopic {index}", ""

    return f"Subtopic {index}", ""


async def _resolve_async_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Phase 1: Planning Nodes
# ---------------------------------------------------------------------------

async def rephrase_topic(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Optimize the user's topic using RephraseAgent."""
    topic = state.get("topic", "")

    # Notify frontend: planning phase started
    await _ws_callback(config, "progress", {
        "status": "planning_started",
        "user_topic": topic,
    })

    pipeline_config = state.get("config", {})
    rephrase_config = pipeline_config.get("planning", {}).get("rephrase", {})

    if not rephrase_config.get("enabled", True):
        await _ws_callback(config, "progress", {
            "status": "rephrase_completed",
            "optimized_topic": topic,
        })
        return {"optimized_topic": topic}

    try:
        from src.agents.research.agents import RephraseAgent

        agent = RephraseAgent(config=pipeline_config)
        result = await agent.process(user_input=topic)

        optimized_topic = result.get("optimized_topic", topic)

        await _ws_callback(config, "progress", {
            "status": "rephrase_completed",
            "optimized_topic": optimized_topic,
        })

        return {"optimized_topic": optimized_topic}

    except Exception as e:
        return {"optimized_topic": topic, "error": str(e)}


async def decompose_topic(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Decompose topic into subtopics using DecomposeAgent."""
    optimized_topic = state.get("optimized_topic", state.get("topic", ""))
    pipeline_config = state.get("config", {})

    decompose_config = pipeline_config.get("planning", {}).get("decompose", {})
    mode = decompose_config.get("mode", "manual")
    num_subtopics = decompose_config.get("initial_subtopics", 5)

    await _ws_callback(config, "progress", {
        "status": "decompose_started",
        "mode": mode,
    })

    try:
        from src.agents.research.agents import DecomposeAgent

        agent = DecomposeAgent(config=pipeline_config, kb_name=state.get("kb_name", "ai_textbook"))
        result = await agent.process(
            topic=optimized_topic,
            num_subtopics=num_subtopics,
            mode=mode,
        )

        sub_topics = result.get("sub_topics", [])

        await _ws_callback(config, "progress", {
            "status": "decompose_completed",
            "generated_subtopics": len(sub_topics),
        })

        return {"sub_topics": sub_topics}

    except Exception as e:
        return {"sub_topics": [], "error": str(e)}


async def initialize_queue(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Initialize the research queue with decomposed topics."""
    sub_topics = state.get("sub_topics", [])
    research_id = state.get("research_id", "")

    from src.agents.research.data_structures import DynamicTopicQueue

    queue = DynamicTopicQueue(research_id=research_id)
    _pipeline_registry[research_id] = {"queue": queue}

    pending_blocks = []
    for index, subtopic in enumerate(sub_topics, start=1):
        sub_topic, overview = _normalize_subtopic_payload(subtopic, index)
        block = queue.add_block(
            sub_topic=sub_topic,
            overview=overview,
        )
        pending_blocks.append(block.block_id)

    # Notify frontend: planning completed with total blocks
    await _ws_callback(config, "progress", {
        "status": "planning_completed",
        "total_blocks": len(pending_blocks),
    })

    return {
        "pending_blocks": pending_blocks,
        "completed_blocks": [],
        "current_block_id": None,
    }


# ---------------------------------------------------------------------------
# Phase 2: Researching Nodes
# ---------------------------------------------------------------------------

async def select_next_block(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Select the next pending block to research."""
    pending_blocks = list(state.get("pending_blocks", []))
    completed_blocks = list(state.get("completed_blocks", []))
    research_id = state.get("research_id", "")

    if not pending_blocks:
        # All blocks done — notify frontend
        await _ws_callback(config, "progress", {
            "status": "researching_completed",
        })
        return {"current_block_id": None}

    next_block_id = pending_blocks[0]

    # On the very first block, transition frontend to researching stage
    if not completed_blocks:
        total = len(pending_blocks) + len(completed_blocks)
        await _ws_callback(config, "progress", {
            "status": "researching_started",
            "total_blocks": total,
            "execution_mode": "series",
        })

    # Get block topic from queue
    registry = _pipeline_registry.get(research_id, {})
    queue = registry.get("queue")
    sub_topic = ""
    if queue:
        for b in queue.blocks:
            if b.block_id == next_block_id:
                sub_topic = b.sub_topic
                break

    # Notify frontend: block started
    await _ws_callback(config, "progress", {
        "status": "block_started",
        "block_id": next_block_id,
        "sub_topic": sub_topic,
        "current_block": len(completed_blocks) + 1,
    })

    return {"current_block_id": next_block_id}


async def research_block(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Research a single topic block."""
    from src.agents.research.data_structures import DynamicTopicQueue, TopicStatus
    from src.agents.research.agents import ResearchAgent, NoteAgent
    from src.agents.research.utils.citation_manager import CitationManager
    from src.tools.rag_tool import rag_search
    from src.tools.web_search import web_search
    from src.tools.query_item_tool import query_numbered_item
    from src.tools.paper_search_tool import PaperSearchTool
    from src.tools.code_executor import run_code
    from src.logging import get_logger

    block_id = state.get("current_block_id")
    research_id = state.get("research_id", "")

    if not block_id:
        return {}

    # Get config
    pipeline_config = state.get("config", {})
    research_config = pipeline_config.get("researching", {})
    rag_config = pipeline_config.get("rag", {})

    max_iterations = research_config.get("max_iterations", 4)
    iteration_mode = research_config.get("iteration_mode", "fixed")
    default_timeout = research_config.get("tool_timeout", 60)

    kb_name = rag_config.get("kb_name", "ai_textbook")

    # Get or create queue (stored in registry for state persistence)
    if research_id not in _pipeline_registry:
        _pipeline_registry[research_id] = {"queue": None}
    registry = _pipeline_registry[research_id]

    if registry["queue"] is None:
        registry["queue"] = DynamicTopicQueue(research_id=research_id)

    if registry.get("citation_manager") is None:
        root_dir = Path(__file__).parent.parent.parent.parent
        cache_dir = root_dir / "data" / "user" / "research" / "cache" / research_id
        registry["citation_manager"] = CitationManager(research_id=research_id, cache_dir=cache_dir)

    queue = registry["queue"]
    citation_manager = registry.get("citation_manager")

    # Get block from queue
    block = None
    for b in queue.blocks:
        if b.block_id == block_id:
            block = b
            break

    if not block:
        print(f"[DEBUG research_block] Block {block_id} not found in queue!")
        return {"error": f"Block {block_id} not found"}

    # Initialize logger
    log_dir = pipeline_config.get("paths", {}).get("user_log_dir") or pipeline_config.get("logging", {}).get("log_dir")
    logger = get_logger("ResearchBlock", log_dir=log_dir)

    try:
        # Mark as researching
        queue.mark_researching(block_id)

        research_agent = ResearchAgent(config=pipeline_config)
        note_agent = NoteAgent(config=pipeline_config)

        # Tool execution function
        async def execute_tool(tool_type: str, query: str) -> str:
            """Execute tool based on type."""
            tool_type = tool_type.lower()

            try:
                if tool_type in ("rag_hybrid", "rag_naive", "rag"):
                    mode = "hybrid" if tool_type == "rag_hybrid" else "naive"
                    result = await rag_search(query=query, kb_name=kb_name, mode=mode)
                    return json.dumps(result, ensure_ascii=False, default=str)

                elif tool_type == "web_search":
                    result = await _resolve_async_result(web_search(query=query, verbose=False))
                    return json.dumps(result, ensure_ascii=False, default=str)

                elif tool_type == "query_item":
                    result = await _resolve_async_result(
                        query_numbered_item(identifier=query, kb_name=kb_name)
                    )
                    return json.dumps(result, ensure_ascii=False, default=str)

                elif tool_type == "paper_search":
                    paper_tool = PaperSearchTool()
                    years_limit = research_config.get("paper_search_years_limit", 3)
                    papers = await _resolve_async_result(
                        paper_tool.search_papers(query=query, max_results=3, years_limit=years_limit)
                    )
                    return json.dumps({"papers": papers}, ensure_ascii=False, default=str)

                elif tool_type == "run_code":
                    result = await run_code(language="python", code=query, timeout=10)
                    return json.dumps(result, ensure_ascii=False, default=str)

                else:
                    # Default to rag_hybrid
                    result = await rag_search(query=query, kb_name=kb_name, mode="hybrid")
                    return json.dumps(result, ensure_ascii=False, default=str)

            except Exception as tool_error:
                logger.warning(f"Tool {tool_type} error: {tool_error}")
                return json.dumps({"status": "failed", "error": str(tool_error)}, ensure_ascii=False)

        # Research loop
        current_knowledge = ""
        iteration = 0

        while iteration < max_iterations:
            # Check if knowledge is sufficient (for flexible mode)
            if iteration_mode == "flexible" and iteration > 0:
                await _ws_callback(config, "progress", {
                    "status": "checking_sufficiency",
                    "block_id": block_id,
                    "iteration": iteration + 1,
                })

                sufficiency = await research_agent.check_sufficiency(
                    topic=block.sub_topic,
                    current_knowledge=current_knowledge,
                )
                if sufficiency.get("is_sufficient", False):
                    await _ws_callback(config, "progress", {
                        "status": "knowledge_sufficient",
                        "block_id": block_id,
                        "reason": sufficiency.get("reason", ""),
                    })
                    break

            # Generate query plan
            await _ws_callback(config, "progress", {
                "status": "generating_query",
                "block_id": block_id,
            })

            query_plan = await research_agent.generate_query_plan(
                topic=block.sub_topic,
                overview=block.overview,
                current_knowledge=current_knowledge,
                iteration=iteration,
            )

            tool_type = query_plan.get("tool_type", "rag_hybrid")
            query = query_plan.get("query", block.sub_topic)
            rationale = query_plan.get("rationale", "")

            # Notify frontend: tool calling
            await _ws_callback(config, "progress", {
                "status": "tool_calling",
                "block_id": block_id,
                "tool_type": tool_type,
                "query": query,
                "rationale": rationale,
            })

            # Execute tool
            raw_answer = await execute_tool(tool_type, query)

            # Notify frontend: tool completed
            await _ws_callback(config, "progress", {
                "status": "tool_completed",
                "block_id": block_id,
                "tool_type": tool_type,
                "query": query,
            })

            # Generate summary with NoteAgent
            await _ws_callback(config, "progress", {
                "status": "processing_notes",
                "block_id": block_id,
            })

            citation_id = f"CIT-{block_id.split('_')[1]}-{iteration + 1:02d}"

            trace = await note_agent.process(
                tool_type=tool_type,
                query=query,
                raw_answer=raw_answer,
                citation_id=citation_id,
            )

            if citation_manager is not None:
                citation_manager.add_citation(
                    citation_id=citation_id,
                    tool_type=tool_type,
                    tool_trace=trace,
                    raw_answer=raw_answer,
                )

            # Add to tool traces
            block.tool_traces.append(trace)

            # Update knowledge
            current_knowledge += f"\n\n{trace.summary}"
            block.iteration_count = iteration + 1

            iteration += 1

        # Mark completed
        queue.mark_completed(block_id)

        await _ws_callback(config, "progress", {
            "status": "block_completed",
            "block_id": block_id,
            "sub_topic": block.sub_topic,
            "iterations": block.iteration_count,
            "tools_used": list({t.tool_type for t in block.tool_traces}),
        })

        return {}

    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.mark_failed(block_id)

        await _ws_callback(config, "progress", {
            "status": "block_failed",
            "block_id": block_id,
            "sub_topic": block.sub_topic,
            "error": str(e),
        })

        return {"error": str(e)}


async def mark_block_complete(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Move current block from pending to completed."""
    pending_blocks = list(state.get("pending_blocks", []))
    completed_blocks = list(state.get("completed_blocks", []))
    current_block_id = state.get("current_block_id")

    if current_block_id and current_block_id in pending_blocks:
        pending_blocks.remove(current_block_id)
        completed_blocks.append(current_block_id)

    await _ws_callback(config, "progress", {
        "status": "block_completed",
        "block_id": current_block_id,
        "completed": len(completed_blocks),
        "remaining": len(pending_blocks),
    })

    return {
        "pending_blocks": pending_blocks,
        "completed_blocks": completed_blocks,
        "current_block_id": None,
    }


# ---------------------------------------------------------------------------
# Phase 3: Reporting Nodes
# ---------------------------------------------------------------------------

async def generate_report(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Generate the final research report using ReportingAgent."""
    from src.agents.research.agents import ReportingAgent

    research_id = state.get("research_id", "")

    # Get queue from registry
    registry = _pipeline_registry.get(research_id, {})
    queue = registry.get("queue")
    citation_manager = registry.get("citation_manager")

    optimized_topic = state.get("optimized_topic", state.get("topic", ""))

    try:
        pipeline_config = state.get("config", {})

        # Progress callback for reporting phase
        async def report_progress_callback(event: dict):
            await _ws_callback(config, "progress", {
                "stage": "reporting",
                **event,
            })

        agent = ReportingAgent(config=pipeline_config)
        if citation_manager is not None:
            agent.set_citation_manager(citation_manager)

        result = await agent.process(
            queue=queue,
            topic=optimized_topic,
            progress_callback=report_progress_callback,
        )

        await _ws_callback(config, "progress", {
            "status": "writing_completed",
            "word_count": result.get("word_count", 0),
        })

        return {"report_result": result}

    except Exception as e:
        return {
            "report_result": {
                "report": f"# Research Report\n\nError generating report: {e}",
                "word_count": 0,
                "error": str(e),
            }
        }


async def save_results(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Save all research results to files."""
    await _ws_callback(config, "progress", {
        "status": "saving_results",
        "message": "Saving research outputs...",
    })

    topic = state.get("topic", "")
    optimized_topic = state.get("optimized_topic", topic)
    report_result = state.get("report_result", {})
    research_id = state.get("research_id", "")

    # Get config for paths
    pipeline_config = state.get("config", {})
    system_config = pipeline_config.get("system", {})

    # Define output directories
    root_dir = Path(__file__).parent.parent.parent.parent
    output_base = root_dir / "data" / "user" / "research"
    cache_dir = output_base / "cache" / research_id
    reports_dir = output_base / "reports"

    # Create directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Get queue from registry for statistics
    registry = _pipeline_registry.get(research_id, {})
    queue = registry.get("queue")

    try:
        # Save report
        report_file = reports_dir / f"{research_id}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_result.get("report", ""))

        # Save queue
        if queue:
            queue_file = cache_dir / "queue.json"
            queue.save_to_json(str(queue_file))

        # Save outline if exists
        if "outline" in report_result:
            outline_file = cache_dir / "outline.json"
            with open(outline_file, "w", encoding="utf-8") as f:
                json.dump(report_result["outline"], f, ensure_ascii=False, indent=2)

        # Get statistics from queue
        statistics = queue.get_statistics() if queue else {}

        # Save metadata
        metadata = {
            "research_id": research_id,
            "topic": topic,
            "optimized_topic": optimized_topic,
            "statistics": statistics,
            "report_word_count": report_result.get("word_count", 0),
            "completed_at": datetime.now().isoformat(),
            "orchestrator": "langgraph",
        }
        metadata_file = reports_dir / f"{research_id}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Token cost statistics
        try:
            from src.agents.research.utils.token_tracker import get_token_tracker

            tracker = get_token_tracker()
            cost_file = cache_dir / "token_cost_summary.json"
            tracker.save(str(cost_file))
        except Exception:
            pass

        await _ws_callback(config, "progress", {
            "status": "research_complete",
            "research_id": research_id,
            "report_path": str(report_file),
        })

        # Cleanup registry
        _pipeline_registry.pop(research_id, None)

        return {
            "result": {
                "research_id": research_id,
                "topic": topic,
                "final_report_path": str(report_file),
                "metadata": metadata,
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "result": {
                "research_id": research_id,
                "error": str(e),
            },
        }


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def check_has_pending(state: ResearchGraphState) -> str:
    """Check if there are pending blocks to research."""
    pending = state.get("pending_blocks", [])
    if pending:
        return "has_pending"
    return "all_complete"


__all__ = [
    "rephrase_topic",
    "decompose_topic",
    "initialize_queue",
    "select_next_block",
    "research_block",
    "mark_block_complete",
    "generate_report",
    "save_results",
    "check_has_pending",
]
