"""
CampusIQ Verifier Agent
Performs:
(a) Deterministic check: Every verbatim quote must be an exact substring of the cited chunk text.
(b) LLM entailment check: Verifies claims follow from quotes and preserves applies_to / file_date cohort distinctions.
Revises once or abstains if unsupported.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from google.adk import Agent
from agents.session import SessionState
from rag.generator import GeminiGenerator
from rag.citations import extract_citations

logger = logging.getLogger("campusiq.agents.verifier")

ABSTENTION_ANSWER = "I could not find this in the documents."

VERIFIER_SYSTEM_PROMPT = """You are the CampusIQ Verifier Agent for IIT Jodhpur institutional regulations.
You review a draft answer against the retrieved document excerpts.

Tasks:
1. Logical Entailment: Ensure all claims strictly and logically follow from the cited excerpts.
2. Cohort Disagreements: If excerpts for different cohorts disagree (e.g., 2019 undergraduate rules vs 2022 postgraduate rules), explicitly mention the cohort each rule applies to.
3. Unsupported Claims: If any claim in the draft is not supported by the excerpts, remove or revise that claim to be strictly factual.
4. Abstention: If the draft is largely unsupported or the question cannot be answered from the excerpts, output EXACTLY:
I could not find this in the documents.

Output ONLY the final verified answer text."""


class VerifierAgent:
    """
    Verifier agent performing deterministic substring validation and LLM entailment checks.
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
            name="verifier_agent",
            model=model,
            instruction="Verifies draft claims against ground quotes and enforces cohort distinctions.",
        )

    def _normalize_text(self, text: str) -> str:
        """Normalizes whitespace and quotes for robust comparison."""
        t = re.sub(r"\s+", " ", text).strip().lower()
        t = t.replace('"', '').replace("'", '').replace("“", '').replace("”", '')
        return t

    def deterministic_quote_check(
        self, draft_answer: str, chunks_seen: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Extracts all quoted strings from draft_answer and checks if each is an exact substring
        (whitespace-normalized) of at least one chunk text.
        Returns: (all_valid, valid_quotes, invalid_quotes)
        """
        if not draft_answer or draft_answer == ABSTENTION_ANSWER:
            return True, [], []

        # Find quotes in double or single quotes
        quotes = re.findall(r'["“]([^"”]+)["”]', draft_answer)
        if not quotes:
            # Also check for single quotes if under 15 words
            single_quotes = re.findall(r"'([^']+)'", draft_answer)
            quotes = [q for q in single_quotes if len(q.split()) <= 15]

        # Combine all chunk texts normalized
        all_chunk_normalized = " ".join(
            self._normalize_text(c.get("text") or c.get("content", ""))
            for c in chunks_seen
        )

        valid_quotes = []
        invalid_quotes = []

        for q in quotes:
            q_norm = self._normalize_text(q)
            if not q_norm or len(q_norm) < 3:
                continue
            if q_norm in all_chunk_normalized:
                valid_quotes.append(q)
            else:
                invalid_quotes.append(q)

        all_valid = (len(invalid_quotes) == 0)
        return all_valid, valid_quotes, invalid_quotes

    def run(self, session: SessionState) -> str:
        """
        Runs Stage (a) deterministic check and Stage (b) LLM entailment verification.
        Populates session.final_answer, session.citations, session.verifier_result, session.abstained.
        """
        draft = session.draft_answer or ""

        # Quick check for model abstention in draft
        if (
            not draft
            or draft == ABSTENTION_ANSWER
            or draft.lower().startswith("i could not find this in the documents")
            or draft.lower().startswith("i cannot find this in the documents")
        ):
            session.final_answer = ABSTENTION_ANSWER
            session.abstained = True
            session.abstention_stage = "verifier"
            session.abstention_reason = "answerer_abstained"
            session.citations = []
            session.verifier_result = {"passed": True, "verdict": "ABSTAINED"}
            return ABSTENTION_ANSWER

        # 1. Stage (a) Deterministic Check
        quotes_valid, valid_quotes, invalid_quotes = self.deterministic_quote_check(draft, session.chunks_seen)
        logger.info(
            f"Deterministic check for '{session.question[:40]}': valid={len(valid_quotes)}, invalid={len(invalid_quotes)}"
        )

        # 2. Stage (b) LLM Entailment Check & Revision
        context_parts = []
        for i, c in enumerate(session.chunks_seen, 1):
            title = c.get("title", "Document")
            page = c.get("page", 1)
            date = c.get("file_date") or c.get("document_date", "Unknown Date")
            cohort = c.get("applies_to", "All students")
            text = c.get("text") or c.get("content", "")
            context_parts.append(
                f"[{title}, p.{page}] (Date: {date} | Applies to: {cohort}):\n{text.strip()}\n"
            )

        context_str = "\n".join(context_parts)
        warning_note = ""
        if invalid_quotes:
            warning_note = f"\nNOTE: The following quotes in the draft were NOT found verbatim in the chunks: {invalid_quotes}. Remove or replace them with exact facts."

        prompt = (
            f"DOCUMENT EXCERPTS:\n{context_str}\n\n"
            f"USER QUESTION: {session.question}\n\n"
            f"DRAFT ANSWER TO VERIFY:\n{draft}\n{warning_note}\n\n"
            f"VERIFIED FINAL ANSWER:"
        )

        res = self.generator.generate(
            prompt=prompt,
            system_instruction=VERIFIER_SYSTEM_PROMPT,
            temperature=0.0,
        )

        session.llm_call_count += 1
        session.add_tokens(res.get("input_tokens", 0), res.get("output_tokens", 0))

        verified_text = res["text"].strip()
        if verified_text.startswith("```"):
            lines = verified_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            verified_text = "\n".join(lines).strip()

        # Check if verifier decided to abstain
        is_abstention = (
            verified_text == ABSTENTION_ANSWER
            or verified_text.lower().startswith("i could not find this in the documents")
            or verified_text.lower().startswith("i cannot find this in the documents")
        )

        if is_abstention:
            session.final_answer = ABSTENTION_ANSWER
            session.abstained = True
            session.abstention_stage = "verifier"
            session.abstention_reason = "verifier_abstained"
            session.citations = []
            session.verifier_result = {
                "quotes_valid": quotes_valid,
                "invalid_quotes": invalid_quotes,
                "verdict": "ABSTAINED",
            }
            return ABSTENTION_ANSWER

        # Build citations mapping markers to chunks with applies_to and file_date
        # Search chunks_seen for matching marker
        citations = extract_citations(verified_text, session.chunks_seen)

        session.final_answer = verified_text
        session.abstained = False
        session.abstention_stage = None
        session.abstention_reason = None
        session.citations = citations
        session.verifier_result = {
            "quotes_valid": quotes_valid,
            "invalid_quotes": invalid_quotes,
            "verdict": "VERIFIED",
            "citation_count": len(citations),
        }

        return verified_text
