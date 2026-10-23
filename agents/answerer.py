"""
CampusIQ Answer Agent
Produces draft answers grounded strictly in retrieved chunks.
Ensures every factual claim includes a short verbatim quote (< 15 words) and [Title, p.X].
Outputs exact abstention string if text is insufficient.
"""

import logging
from typing import Any, Dict, List, Optional
from google.adk import Agent
from agents.session import SessionState
from rag.generator import GeminiGenerator

logger = logging.getLogger("campusiq.agents.answerer")

ABSTENTION_ANSWER = "I could not find this in the documents."

ANSWERER_SYSTEM_PROMPT = """You are the CampusIQ Grounded Answer Agent for IIT Jodhpur institutional regulations.
Answer the question using ONLY the provided document excerpts.

Strict Requirements:
1. Every factual statement or claim MUST be immediately supported by a short verbatim quote (under 15 words in double quotes) taken directly from the excerpt text, followed by its source marker [Title, p.X].
   Example: Under the regulations, "minimum attendance of 75% in all lectures" is required [Academic Regulations, p.7].
2. Do NOT extrapolate, speculate, or introduce any external knowledge.
3. If the provided excerpts do not directly state the answer to the question, you MUST respond with EXACTLY:
I could not find this in the documents.
4. Keep the answer direct, factual, and concise."""


class AnswererAgent:
    """
    Generates draft answers with verbatim quotes and citations.
    """

    def __init__(
        self,
        generator: Optional[GeminiGenerator] = None,
        model: str = "gemini-3.5-flash-lite",
    ):
        self.generator = generator or GeminiGenerator(model=model)
        self.model = model

        # Google ADK Agent
        self.adk_agent = Agent(
            name="answerer_agent",
            model=model,
            instruction="Generates strictly grounded answers with verbatim quotes and citations.",
        )

    def run(self, session: SessionState) -> str:
        """
        Produces draft answer from session.chunks_seen.
        Updates session.draft_answer and token counts.
        """
        if not session.chunks_seen:
            session.draft_answer = ABSTENTION_ANSWER
            return ABSTENTION_ANSWER

        # Format chunks as context
        context_parts = []
        for i, c in enumerate(session.chunks_seen, 1):
            title = c.get("title", "Document")
            page = c.get("page", 1)
            date = c.get("file_date") or c.get("document_date", "Unknown Date")
            cohort = c.get("applies_to", "All students")
            text = c.get("text") or c.get("content", "")
            context_parts.append(
                f"--- Excerpt {i} [{title}, p.{page}] (Date: {date} | Applies to: {cohort}) ---\n{text.strip()}\n"
            )

        context_str = "\n".join(context_parts)
        prompt = f"DOCUMENT EXCERPTS:\n{context_str}\n\nQUESTION: {session.question}\n\nDRAFT ANSWER:"

        res = self.generator.generate(
            prompt=prompt,
            system_instruction=ANSWERER_SYSTEM_PROMPT,
            temperature=0.0,
        )

        session.llm_call_count += 1
        session.add_tokens(res.get("input_tokens", 0), res.get("output_tokens", 0))

        draft = res["text"].strip()
        # Clean any markdown block wrapping if present
        if draft.startswith("```"):
            lines = draft.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            draft = "\n".join(lines).strip()

        session.draft_answer = draft
        return draft
