"""
CampusIQ RAG Prompt Templates
Defines strict system instructions and context formatting for grounded academic QA.
"""

from typing import List
from retrieval.vector_store import SearchResult

SYSTEM_PROMPT = """You are CampusIQ, the official academic regulations and curriculum assistant for IIT Jodhpur.
Your duty is to provide strictly factual, grounded, and verified answers based ONLY on the provided excerpts from official institute documents.

STRICT INSTRUCTIONS:
1. Grounding: Answer ONLY based on the provided context chunks. Do not use any outside knowledge, assumptions, or extrapolations.
2. Citation Requirement: Support every factual claim by citing the source document using the format: [Document Title, p.X] (or the exact citation provided with the chunk).
3. Abstention Rule: If the provided chunks do not contain sufficient evidence to answer the question, or if the question asks for information not present in the excerpts, you MUST reply with EXACTLY and ONLY:
   "I could not find this in the documents."
4. Conciseness: Provide direct, clear, and comprehensive answers without unnecessary preamble.
5. No Fabrication: Never guess dates, CGPA requirements, penalties, rules, or courses.
"""

USER_TEMPLATE = """Context Chunks from Official IIT Jodhpur Documents:
{context_blocks}

User Question: {question}

Answer (citing sources as [Title, p.X], or reply exactly "I could not find this in the documents."):"""


def format_context_blocks(chunks: List[SearchResult]) -> str:
    """
    Formats retrieved search results into clear numbered context blocks for LLM ingestion.
    """
    blocks = []
    for idx, c in enumerate(chunks, start=1):
        citation = c.to_citation()
        block = (
            f"--- Chunk {idx}: {citation} (Similarity Score: {c.score:.4f}) ---\n"
            f"{c.content.strip()}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def build_prompt(question: str, chunks: List[SearchResult]) -> str:
    """
    Builds the complete grounded QA prompt for the generator.
    """
    context_blocks = format_context_blocks(chunks)
    return USER_TEMPLATE.format(context_blocks=context_blocks, question=question.strip())
