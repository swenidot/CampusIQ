"""
CampusIQ RAG Package
Exposes RAGPipeline, GeminiGenerator, and prompt builders.
"""

from rag.qa import RAGPipeline, ABSTENTION_ANSWER, DEFAULT_SIMILARITY_THRESHOLD
from rag.generator import GeminiGenerator
from rag.prompt import SYSTEM_PROMPT, build_prompt

__all__ = [
    "RAGPipeline",
    "GeminiGenerator",
    "SYSTEM_PROMPT",
    "build_prompt",
    "ABSTENTION_ANSWER",
    "DEFAULT_SIMILARITY_THRESHOLD",
]
