"""
CampusIQ Verifier Agent (v2)
Performs:
(a) Deterministic check: Normalized exact substring check with line-break repair and punctuation stripping.
(b) Fuzzy guard check: Every number, percentage, and date must be present, plus >= 90% non-stopword tokens.
(c) Non-destructive pass: Approved drafts are returned untouched without rewriting, preserving all citation markers.
(d) Entailment revision: Only unsupported claims are revised or removed.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from google.adk import Agent
from agents.session import SessionState
from rag.generator import GeminiGenerator
from rag.citations import extract_citations

logger = logging.getLogger("campusiq.agents.verifier")

ABSTENTION_ANSWER = "I could not find this in the documents."

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "in", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "shall", "per", "as", "it", "its"
}

MONTH_TOKENS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"
}

VERIFIER_REVISION_PROMPT = """You are the CampusIQ Verifier Agent for IIT Jodhpur institutional regulations.
A draft answer contained factual claims whose supporting quotes failed verification against the excerpts.

Tasks:
1. Revise the answer to remove or correct only the unsupported claims.
2. PRESERVE ALL [Title, p.X] citation markers for verified claims. Do NOT strip or change citation markers.
3. If the entire answer is unsupported, output EXACTLY:
I could not find this in the documents.

Output ONLY the revised answer text."""


class VerifierAgent:
    """
    Verifier agent performing normalized substring validation, fuzzy numeric guards,
    and non-destructive finalization.
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
        """
        Normalizes quote and chunk text:
        - lowercase
        - repair line-break hyphenation (e.g., 'semes-\\nter' -> 'semester')
        - replace newlines and tabs with spaces
        - strip LaTeX commands and symbols
        - strip punctuation, preserving alphanumeric, % and decimal numbers
        - collapse multiple spaces
        """
        if not text:
            return ""
        t = text.lower()
        # Repair line-break hyphenation
        t = re.sub(r"(\w+)-\s*[\r\n]+\s*(\w+)", r"\1\2", t)
        # Replace newlines and tabs with spaces
        t = re.sub(r"[\r\n\t]+", " ", t)
        # Strip LaTeX commands and symbols
        t = re.sub(r"\\[a-zA-Z]+", " ", t)
        t = re.sub(r"[\$\\\{\}\^]+", " ", t)
        # Strip punctuation except % and decimal points in numbers
        t = re.sub(r"[^\w\s%\.]", " ", t)
        # Strip standalone periods
        t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def verify_quote(self, quote: str, all_chunk_normalized: str) -> Tuple[bool, str]:
        """
        Verifies a single quote against all_chunk_normalized.
        Returns: (passed, path) where path is 'exact_normalized', 'fuzzy', or 'failed'.
        """
        q_norm = self._normalize_text(quote)
        if not q_norm or len(q_norm) < 2:
            return True, "exact_normalized"

        # (1) Normalized exact substring check
        if q_norm in all_chunk_normalized:
            return True, "exact_normalized"

        # (2) Fuzzy verification check
        # (i) Every number, percentage, and date token in quote MUST appear in chunk
        num_pct_tokens = re.findall(r"\b\d+(?:\.\d+)?%?\b", q_norm)
        date_tokens = [w for w in q_norm.split() if w in MONTH_TOKENS]
        guarded_tokens = num_pct_tokens + date_tokens

        for token in guarded_tokens:
            pattern = r"\b" + re.escape(token) + r"\b"
            if not re.search(pattern, all_chunk_normalized):
                return False, "failed"

        # (ii) At least 90% of non-stopword tokens in quote must appear in chunk
        tokens = [w for w in q_norm.split() if w not in STOPWORDS and len(w) > 1 and w not in guarded_tokens]
        if tokens:
            matched = [w for w in tokens if re.search(r"\b" + re.escape(w) + r"\b", all_chunk_normalized)]
            ratio = len(matched) / len(tokens)
            if ratio < 0.90:
                return False, "failed"

        return True, "fuzzy"

    def deterministic_quote_check(
        self, draft_answer: str, chunks_seen: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Extracts all quoted snippets from draft_answer and validates each.
        Returns: (all_valid, valid_quotes, invalid_quotes)
        Sets self.last_paths_used with the verification path for each quote.
        """
        if not draft_answer or draft_answer == ABSTENTION_ANSWER:
            self.last_paths_used = []
            return True, [], []

        # Find quotes in double or single quotes
        quotes = re.findall(r'["“]([^"”]+)["”]', draft_answer)
        if not quotes:
            single_quotes = re.findall(r"'([^']+)'", draft_answer)
            quotes = [q for q in single_quotes if len(q.split()) <= 15]

        all_chunk_normalized = " ".join(
            self._normalize_text(c.get("text") or c.get("content", ""))
            for c in chunks_seen
        )

        valid_quotes = []
        invalid_quotes = []
        paths_used = []

        if not quotes:
            self.last_paths_used = ["exact_normalized"]
            return True, [], []

        for q in quotes:
            passed, path = self.verify_quote(q, all_chunk_normalized)
            if passed:
                valid_quotes.append(q)
                paths_used.append(path)
            else:
                invalid_quotes.append(q)
                paths_used.append("failed")

        self.last_paths_used = paths_used
        all_valid = (len(invalid_quotes) == 0)
        return all_valid, valid_quotes, invalid_quotes

    def run(self, session: SessionState) -> str:
        """
        Runs Stage (a) deterministic/fuzzy check and Stage (b) non-destructive finalization.
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
            session.verification_path = "failed"
            session.verifier_result = {"passed": True, "verdict": "ABSTAINED"}
            return ABSTENTION_ANSWER

        # 1. Deterministic & Fuzzy Quote Validation
        quotes_valid, valid_quotes, invalid_quotes = self.deterministic_quote_check(
            draft, session.chunks_seen
        )
        paths_used = getattr(self, "last_paths_used", ["exact_normalized"])
        for p in paths_used:
            session.verification_paths.append(p)

        dominant_path = (
            "failed" if "failed" in paths_used else
            ("fuzzy" if "fuzzy" in paths_used else "exact_normalized")
        )
        session.verification_path = dominant_path

        # 2. Non-Destructive Pass: if all quotes pass, do NOT rewrite with LLM
        if quotes_valid:
            citations = extract_citations(draft, session.chunks_seen)
            session.final_answer = draft
            session.abstained = False
            session.abstention_stage = None
            session.abstention_reason = None
            session.citations = citations
            session.verifier_result = {
                "quotes_valid": True,
                "invalid_quotes": [],
                "path": dominant_path,
                "verdict": "VERIFIED",
                "citation_count": len(citations),
            }
            logger.info(
                f"Non-destructive pass for '{session.question[:40]}': path={dominant_path}, citations={len(citations)}"
            )
            return draft

        # 3. Fallback Revision: If one or more quotes failed, revise once via LLM
        logger.warning(
            f"Quote check failed for '{session.question[:40]}': invalid={invalid_quotes}. Triggering verifier revision."
        )

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
        prompt = (
            f"DOCUMENT EXCERPTS:\n{context_str}\n\n"
            f"USER QUESTION: {session.question}\n\n"
            f"DRAFT ANSWER TO REVISE:\n{draft}\n\n"
            f"UNVERIFIED QUOTES TO FIX OR REMOVE: {invalid_quotes}\n\n"
            f"REVISED FINAL ANSWER:"
        )

        t_llm_start = time.perf_counter()
        res = self.generator.generate(
            prompt=prompt,
            system_instruction=VERIFIER_REVISION_PROMPT,
            temperature=0.0,
        )
        llm_elapsed_ms = (time.perf_counter() - t_llm_start) * 1000
        session.compute_ms += llm_elapsed_ms

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
                "quotes_valid": False,
                "invalid_quotes": invalid_quotes,
                "path": "failed",
                "verdict": "ABSTAINED",
            }
            return ABSTENTION_ANSWER

        citations = extract_citations(verified_text, session.chunks_seen)
        session.final_answer = verified_text
        session.abstained = False
        session.abstention_stage = None
        session.abstention_reason = None
        session.citations = citations
        session.verifier_result = {
            "quotes_valid": False,
            "invalid_quotes": invalid_quotes,
            "path": "revised",
            "verdict": "REVISED",
            "citation_count": len(citations),
        }

        return verified_text
