# -*- coding: utf-8 -*-
"""
Question Graph Nodes
=====================

Node functions for LangGraph question generation workflows.
Each node wraps an existing agent's process() method and returns state updates.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from .state import QuestionGraphState


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


# ---------------------------------------------------------------------------
# Stage 1: Retrieve
# ---------------------------------------------------------------------------

async def retrieve(state: QuestionGraphState, config: RunnableConfig) -> dict:
    """Retrieve knowledge from the knowledge base using RetrieveAgent."""
    from src.agents.question.agents.retrieve_agent import RetrieveAgent

    await _ws_callback(config, "progress", {
        "stage": "researching", "progress": {"status": "retrieving"},
    })

    agent = RetrieveAgent(
        kb_name=state.get("kb_name"),
        language=state.get("language", "en"),
    )

    result = await agent.process(
        requirement=state["requirement"],
        num_queries=state.get("rag_query_count", 3),
    )

    has_content = result.get("has_content", False)

    if has_content:
        await _ws_callback(config, "knowledge_saved", {
            "queries": result.get("queries", []),
        })

    return {
        "retrieval_result": result,
        "knowledge_context": result.get("summary", ""),
        "has_content": has_content,
        "queries": result.get("queries", []),
    }


# ---------------------------------------------------------------------------
# Stage 2: Plan (custom mode)
# ---------------------------------------------------------------------------

async def plan(state: QuestionGraphState, config: RunnableConfig) -> dict:
    """Generate a question plan with distinct focuses using direct LLM call."""
    from src.services.llm import complete as llm_complete
    from src.services.llm.config import get_llm_config

    await _ws_callback(config, "progress", {
        "stage": "planning", "progress": {"status": "creating_plan"},
    })

    requirement = state["requirement"]
    knowledge_context = state["knowledge_context"]
    num_questions = state.get("num_questions", 1)

    llm_config = get_llm_config()

    system_prompt = (
        "You are an educational content planner. Create distinct question focuses "
        "that test different aspects of the same topic.\n\n"
        "CRITICAL: Return ONLY valid JSON. Do not wrap in markdown code blocks.\n"
        'Output JSON with key "focuses" containing an array of objects, each with:\n'
        '- "id": string like "q_1", "q_2"\n'
        '- "focus": string describing what aspect to test\n'
        f'- "type": "{requirement.get("question_type", "written")}"'
    )

    truncated = knowledge_context[:4000]
    suffix = "...[truncated]" if len(knowledge_context) > 4000 else ""

    user_prompt = (
        f"Topic: {requirement.get('knowledge_point', '')}\n"
        f"Difficulty: {requirement.get('difficulty', 'medium')}\n"
        f"Question Type: {requirement.get('question_type', 'written')}\n"
        f"Number: {num_questions}\n\n"
        f"Knowledge:\n{truncated}{suffix}\n\n"
        f"Generate exactly {num_questions} distinct focuses in JSON."
    )

    try:
        response = await llm_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            api_version=getattr(llm_config, "api_version", None),
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(response)
        focuses = data.get("focuses", [])
        if not isinstance(focuses, list):
            focuses = []
    except Exception:
        focuses = []

    # Fallback if planning failed
    if len(focuses) < num_questions:
        for i in range(len(focuses), num_questions):
            focuses.append({
                "id": f"q_{i + 1}",
                "focus": f"Aspect {i + 1} of {requirement.get('knowledge_point', 'the topic')}",
                "type": requirement.get("question_type", "written"),
            })

    plan_result = {"focuses": focuses[:num_questions]}

    await _ws_callback(config, "plan_ready", {
        "plan": plan_result, "focuses": focuses[:num_questions],
    })

    return {
        "plan": plan_result,
        "focuses": focuses[:num_questions],
        "current_focus_index": 0,
        "results": [],
        "failures": [],
    }


# ---------------------------------------------------------------------------
# Stage 3: Generate + Analyze (one question at a time)
# ---------------------------------------------------------------------------

async def generate_and_analyze(state: QuestionGraphState, config: RunnableConfig) -> dict:
    """Generate one question and analyze its relevance."""
    from src.agents.question.agents.generate_agent import GenerateAgent
    from src.agents.question.agents.relevance_analyzer import RelevanceAnalyzer

    idx = state.get("current_focus_index", 0)
    focuses = state.get("focuses", [])
    results = list(state.get("results", []))
    failures = list(state.get("failures", []))
    num_questions = state.get("num_questions", len(focuses))

    if idx >= len(focuses):
        return {"current_focus_index": idx}

    focus = focuses[idx]
    question_id = focus.get("id", f"q_{idx + 1}")

    await _ws_callback(config, "question_update", {
        "question_id": question_id, "status": "generating",
        "focus": focus.get("focus", ""),
    })

    # Generate
    gen_agent = GenerateAgent(language=state.get("language", "en"))
    gen_result = await gen_agent.process(
        requirement=state["requirement"],
        knowledge_context=state["knowledge_context"],
        focus=focus,
    )

    if not gen_result.get("success"):
        failures.append({
            "question_id": question_id,
            "error": gen_result.get("error", "Unknown error"),
        })
        await _ws_callback(config, "question_update", {
            "question_id": question_id, "status": "error",
        })
        return {
            "current_focus_index": idx + 1,
            "results": results,
            "failures": failures,
        }

    question = gen_result["question"]

    # Analyze relevance
    await _ws_callback(config, "question_update", {
        "question_id": question_id, "status": "analyzing",
    })

    analyzer = RelevanceAnalyzer(language=state.get("language", "en"))
    analysis = await analyzer.process(
        question=question,
        knowledge_context=state["knowledge_context"],
    )

    validation = {
        "decision": "approve",
        "relevance": analysis["relevance"],
        "kb_coverage": analysis["kb_coverage"],
        "extension_points": analysis.get("extension_points", ""),
    }

    result = {
        "question_id": question_id,
        "focus": focus,
        "question": question,
        "analysis": analysis,
        "validation": validation,
    }
    results.append(result)

    await _ws_callback(config, "question_update", {
        "question_id": question_id, "status": "done",
    })
    await _ws_callback(config, "result", {
        "question_id": question_id,
        "question": question,
        "validation": validation,
        "focus": focus,
        "index": idx,
    })
    await _ws_callback(config, "progress", {
        "stage": "generating",
        "progress": {"current": idx + 1, "total": num_questions},
    })

    return {
        "current_focus_index": idx + 1,
        "results": results,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Single-question mode: generate + analyze in one node
# ---------------------------------------------------------------------------

async def generate_single(state: QuestionGraphState, config: RunnableConfig) -> dict:
    """Generate a single question and analyze relevance (for single-question mode)."""
    from src.agents.question.agents.generate_agent import GenerateAgent
    from src.agents.question.agents.relevance_analyzer import RelevanceAnalyzer

    await _ws_callback(config, "progress", {
        "stage": "generating", "progress": {"status": "initializing"},
    })

    requirement = state["requirement"]
    knowledge_context = state["knowledge_context"]
    reference_question = requirement.get("reference_question")

    # Generate
    gen_agent = GenerateAgent(language=state.get("language", "en"))
    gen_result = await gen_agent.process(
        requirement=requirement,
        knowledge_context=knowledge_context,
        reference_question=reference_question,
    )

    if not gen_result.get("success"):
        return {
            "error": gen_result.get("error", "Generation failed"),
            "results": [],
        }

    question = gen_result["question"]

    # Analyze
    analyzer = RelevanceAnalyzer(language=state.get("language", "en"))
    analysis = await analyzer.process(
        question=question,
        knowledge_context=knowledge_context,
    )

    result = {
        "success": True,
        "question": question,
        "validation": {
            "decision": "approve",
            "relevance": analysis["relevance"],
            "kb_coverage": analysis["kb_coverage"],
            "extension_points": analysis.get("extension_points", ""),
        },
        "rounds": 1,
    }

    # Send result to frontend via WebSocket
    await _ws_callback(config, "result", {
        "question_id": "q_1",
        "question": question,
        "validation": result["validation"],
        "focus": {"id": "q_1", "focus": requirement.get("knowledge_point", "")},
        "index": 0,
    })
    await _ws_callback(config, "progress", {
        "stage": "complete",
        "completed": 1,
        "failed": 0,
        "total": 1,
    })

    return {
        "results": [result],
        "summary": result,
    }


# ---------------------------------------------------------------------------
# Build summary (custom mode)
# ---------------------------------------------------------------------------

async def build_summary(state: QuestionGraphState, config: RunnableConfig) -> dict:
    """Build final summary for custom question generation."""
    results = state.get("results", [])
    failures = state.get("failures", [])
    num_questions = state.get("num_questions", 0)
    queries = state.get("queries", [])
    plan_result = state.get("plan", {})

    summary = {
        "success": len(results) == num_questions,
        "requested": num_questions,
        "completed": len(results),
        "failed": len(failures),
        "search_queries": queries,
        "plan": plan_result,
        "results": results,
        "failures": failures,
    }

    # Save to output_dir if specified
    output_dir = state.get("output_dir")
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = Path(output_dir) / f"batch_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        with open(batch_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["output_dir"] = str(batch_dir)

    await _ws_callback(config, "progress", {
        "stage": "complete",
        "completed": len(results),
        "failed": len(failures),
        "total": num_questions,
    })

    return {"summary": summary}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def check_has_content(state: QuestionGraphState) -> str:
    """Route based on whether knowledge content was found."""
    if state.get("has_content"):
        return "has_content"
    return "no_content"


def check_mode(state: QuestionGraphState) -> str:
    """Route based on single vs custom (multi-question) mode."""
    if state.get("num_questions", 1) > 1:
        return "custom"
    return "single"


def check_more_focuses(state: QuestionGraphState) -> str:
    """Check if there are more focuses to generate."""
    idx = state.get("current_focus_index", 0)
    focuses = state.get("focuses", [])
    if idx < len(focuses):
        return "more"
    return "done"


__all__ = [
    "retrieve",
    "plan",
    "generate_and_analyze",
    "generate_single",
    "build_summary",
    "check_has_content",
    "check_mode",
    "check_more_focuses",
]
