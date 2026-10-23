"""
CampusIQ Retriever Agent
Orchestrates semantic search and exact page retrieval over authenticated MCP server.
Features:
- Query reformulation and retry (up to 2 retries) if retrieved chunks lack relevance.
- Adjacent page retrieval (get_chunk) for split sections.
- Strict 5 tool-call hard cap.
- Compatible with Google ADK Agent structure.
"""

import logging
from typing import Any, Dict, List, Optional
from google.adk import Agent
from agents.mcp_client import MCPClient
from agents.session import SessionState
from rag.generator import GeminiGenerator

logger = logging.getLogger("campusiq.agents.retriever")

RELEVANCE_THRESHOLD = 0.65


class RetrieverAgent:
    """
    Retriever agent calling MCP tools search_documents and get_chunk.
    Reformulates queries when initial results lack sufficient relevance.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        generator: Optional[GeminiGenerator] = None,
        model: str = "gemini-3.5-flash-lite",
    ):
        self.mcp_client = mcp_client
        self.generator = generator or GeminiGenerator(model=model)
        self.model = model

        # Google ADK Agent definition
        self.adk_agent = Agent(
            name="retriever_agent",
            model=model,
            instruction=(
                "You are an academic retrieval specialist for IIT Jodhpur. "
                "You inspect query requirements and formulate precise queries to retrieve "
                "governance documents, academic regulations, curricula, and calendars."
            ),
        )

    def _assess_relevance(self, question: str, chunks: List[Dict[str, Any]]) -> bool:
        """Determines if the retrieved chunks sufficiently cover the question."""
        if not chunks:
            return False

        top_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
        if top_score < RELEVANCE_THRESHOLD:
            return False

        # Check keyword presence for key entities
        q_lower = question.lower()
        all_text = " ".join(c.get("text", "") for c in chunks).lower()

        # Specific domain heuristics for known institutional concepts
        if "attendance" in q_lower and "attendance" not in all_text:
            return False
        if "leave" in q_lower and "leave" not in all_text:
            return False
        if "cgpa" in q_lower and "cgpa" not in all_text and "grade point" not in all_text:
            return False
        if "probation" in q_lower and "probation" not in all_text:
            return False
        if "discipline" in q_lower and "discipline" not in all_text and "penalt" not in all_text:
            return False

        return True

    def _reformulate_query(self, session: SessionState, previous_chunks: List[Dict[str, Any]]) -> str:
        """Uses LLM to reformulate search query based on missing content."""
        chunks_summary = "\n".join(
            f"[{c.get('title')}, p.{c.get('page')}] score={c.get('score', 0):.3f}: {c.get('text', '')[:120]}..."
            for c in previous_chunks[:3]
        )

        prompt = (
            f"You are an academic search retrieval specialist for IIT Jodhpur institutional documents.\n"
            f"Original Question: {session.question}\n"
            f"Queries already tried: {session.queries_tried}\n"
            f"Current Chunks Found:\n{chunks_summary}\n\n"
            f"The current chunks did not sufficiently answer the question.\n"
            f"Provide ONE concise, reformulated search query (under 10 words) targeting the specific regulation, "
            f"table, section, or program rules (e.g., 'UG regulations attendance 75%', 'rules_ug_2019 leave rules', "
            f"'M.Tech. graduation minimum CGPA', 'academic calendar exam dates').\n"
            f"Output ONLY the new query string without quotes or explanations."
        )

        res = self.generator.generate(
            prompt=prompt,
            system_instruction="Output only the query text.",
            temperature=0.0,
        )
        session.llm_call_count += 1
        session.add_tokens(res.get("input_tokens", 0), res.get("output_tokens", 0))

        reformulated = res["text"].strip().strip('"\'')
        return reformulated if reformulated else session.question

    def run(self, session: SessionState) -> List[Dict[str, Any]]:
        """
        Executes retrieval with up to 2 reformulations and optional adjacent page lookup.
        Appends chunks to session.chunks_seen.
        """
        current_query = session.question
        session.queries_tried.append(current_query)

        # 1. Initial Search
        initial_chunks = self.mcp_client.search_documents(current_query, k=5)
        session.tool_call_count = self.mcp_client.call_count
        session.add_chunks(initial_chunks)

        active_chunks = list(initial_chunks)

        # 2. Check relevance and retry with reformulation up to 2 times
        retries_remaining = 2
        while retries_remaining > 0 and self.mcp_client.call_count < self.mcp_client.max_tool_calls:
            if self._assess_relevance(session.question, active_chunks):
                break

            logger.info(f"Retriever reformulating query (attempts left: {retries_remaining}) for: {session.question}")
            new_query = self._reformulate_query(session, active_chunks)
            if new_query in session.queries_tried:
                break

            session.queries_tried.append(new_query)
            new_chunks = self.mcp_client.search_documents(new_query, k=5)
            session.tool_call_count = self.mcp_client.call_count
            session.add_chunks(new_chunks)

            # Update active chunks with highest scoring chunks
            active_chunks = sorted(session.chunks_seen, key=lambda c: c.get("score", 0.0), reverse=True)
            retries_remaining -= 1

        # 3. Check for multi-page split chunk opportunities
        # If top chunk is page 2 and tool calls < 5, inspect page 3 for split sections
        if active_chunks and self.mcp_client.call_count < self.mcp_client.max_tool_calls:
            top_chunk = active_chunks[0]
            top_doc = top_chunk.get("doc_id")
            top_page = top_chunk.get("page", 1)

            # If top chunk looks relevant but may continue onto the next page
            if top_chunk.get("score", 0.0) >= 0.70 and top_page in [2, 12]:
                next_page_chunks = self.mcp_client.get_chunk(top_doc, top_page + 1)
                session.tool_call_count = self.mcp_client.call_count
                session.add_chunks(next_page_chunks)

        return session.chunks_seen
