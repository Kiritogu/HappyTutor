"""
Question Generation System

Modular question generation using specialized agents:
- RetrieveAgent: Knowledge base retrieval
- GenerateAgent: Question generation
- RelevanceAnalyzer: Question-KB relevance analysis
- AgentCoordinator: CLI workflow orchestration (legacy)

Orchestration via LangGraph (src/agents/question/graph) is preferred for API usage.

Tools (moved to src/tools/question):
- parse_pdf_with_mineru
- extract_questions_from_paper
- mimic_exam_questions
"""

from .agents import GenerateAgent, RelevanceAnalyzer, RetrieveAgent
from .coordinator import AgentCoordinator

__all__ = [
    "RetrieveAgent",
    "GenerateAgent",
    "RelevanceAnalyzer",
    "AgentCoordinator",
]
