# -*- coding: utf-8 -*-
"""
Question Generation Graph
==========================

LangGraph-based orchestration for question generation workflows.

Usage:
    from src.agents.question.graph import build_question_graph, QuestionGraphState

    graph = build_question_graph()
    result = await graph.ainvoke(
        {
            "requirement": {"knowledge_point": "...", "difficulty": "medium"},
            "kb_name": "my_kb",
            "language": "zh",
            "num_questions": 3,
        },
        config={"configurable": {"thread_id": "task_123"}},
    )
"""

from .graph import build_question_graph
from .state import QuestionGraphState

__all__ = [
    "build_question_graph",
    "QuestionGraphState",
]
