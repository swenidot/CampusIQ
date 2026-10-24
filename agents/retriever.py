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

import time

logger = logging.getLogger("campusiq.agents.retriever")

RELEVANCE_THRESHOLD = 0.65


class RetrieverAgent:
    """
    Retriever agent calling MCP tools search_documents and get_chunk.
    Implements baseline-first retrieval, unanswerable fast-exit,
    deterministic top-2 adjacent-chunk expansion, and targeted retry with k=8.
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

    def _reformulate_query(self, session: SessionState, previous_chunks: List[Dict[str, Any]]) -> str:
        """Uses LLM to reformulate search query based on missing content."""
        chunks_summary = "\n".join(
            f"[{c.get('title')}, p.{c.get('page')}] score={c.get('score', 0):.3f}: {(c.get('text') or c.get('content') or '')[:120]}..."
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

        t_llm_start = time.perf_counter()
        res = self.generator.generate(
            prompt=prompt,
            system_instruction="Output only the query text.",
            temperature=0.0,
        )
        elapsed_ms = (time.perf_counter() - t_llm_start) * 1000
        session.compute_ms += elapsed_ms

        session.llm_call_count += 1
        session.add_tokens(res.get("input_tokens", 0), res.get("output_tokens", 0))

        reformulated = res["text"].strip().strip('"\'')
        return reformulated if reformulated else session.question

    def initial_retrieval(self, session: SessionState) -> List[Dict[str, Any]]:
        """
        Executes baseline-first retrieval:
        1. Single direct search_documents call with k=5.
        2. Fast-exit if top-1 score < 0.65 (unanswerable questions exit with 0 LLM calls).
        3. Deterministic top-2 adjacent-chunk expansion via get_chunk.
        4. Cap total chunks at 8.
        """
        current_query = session.question
        session.queries_tried.append(current_query)

        # 1. Initial Search with k=5
        t_mcp_start = time.perf_counter()
        initial_chunks = self.mcp_client.search_documents(current_query, k=5)
        mcp_ms = (time.perf_counter() - t_mcp_start) * 1000
        session.compute_ms += mcp_ms
        session.tool_call_count = self.mcp_client.call_count

        if not initial_chunks:
            session.abstained = True
            session.abstention_stage = "stage_1_threshold"
            session.abstention_reason = "threshold_abstention"
            session.final_answer = "I could not find this in the documents."
            return []

        top_score = max((c.get("score", 0.0) for c in initial_chunks), default=0.0)
        session.top_score = round(top_score, 4)

        # 2. Fast-exit if below Stage-1 similarity threshold (0.65)
        if top_score < RELEVANCE_THRESHOLD:
            logger.info(f"Fast-exit threshold abstention (top_score={top_score:.4f} < {RELEVANCE_THRESHOLD})")
            session.abstained = True
            session.abstention_stage = "stage_1_threshold"
            session.abstention_reason = "threshold_abstention"
            session.final_answer = "I could not find this in the documents."
            session.add_chunks(initial_chunks)
            return initial_chunks

        # Add initial chunks
        session.add_chunks(initial_chunks)

        # 3. Deterministic Adjacent-Chunk Expansion for Top-2 Ranked Chunks
        top_2_chunks = initial_chunks[:2]
        for chunk in top_2_chunks:
            if self.mcp_client.call_count >= self.mcp_client.max_tool_calls:
                break
            doc_id = chunk.get("doc_id")
            page = chunk.get("page", 1)
            if not doc_id:
                continue

            # Fetch immediate continuation (next page)
            from observability.tracing import trace_span
            with trace_span("adjacent_expansion", {"tool_name": "get_chunk", "page": page + 1}):
                t_adj_start = time.perf_counter()
                next_page_chunks = self.mcp_client.get_chunk(doc_id=doc_id, page=page + 1)
                adj_ms = (time.perf_counter() - t_adj_start) * 1000
                session.compute_ms += adj_ms
                session.tool_call_count = self.mcp_client.call_count

                if next_page_chunks:
                    session.add_chunks(next_page_chunks)

        # Cap total chunks passed to answerer at 8
        if len(session.chunks_seen) > 8:
            session.chunks_seen = session.chunks_seen[:8]

        return session.chunks_seen

    def targeted_retry(self, session: SessionState) -> List[Dict[str, Any]]:
        """
        Executes targeted retry with one reformulated search (k=8).
        Only invoked when initial retrieval had top-score >= 0.65 but answerer abstained.
        """
        if self.mcp_client.call_count >= self.mcp_client.max_tool_calls:
            logger.warning("Targeted retry skipped: Hard cap of 5 tool calls reached.")
            return session.chunks_seen

        new_query = self._reformulate_query(session, session.chunks_seen)
        if new_query in session.queries_tried:
            logger.info("Targeted retry skipped: Reformulated query already tried.")
            return session.chunks_seen

        session.queries_tried.append(new_query)
        logger.info(f"Targeted retry with query '{new_query}' (k=8)")

        t_mcp_start = time.perf_counter()
        new_chunks = self.mcp_client.search_documents(new_query, k=8)
        mcp_ms = (time.perf_counter() - t_mcp_start) * 1000
        session.compute_ms += mcp_ms
        session.tool_call_count = self.mcp_client.call_count

        if new_chunks:
            session.add_chunks(new_chunks)

        if len(session.chunks_seen) > 8:
            session.chunks_seen = session.chunks_seen[:8]

        return session.chunks_seen

    def run(self, session: SessionState) -> List[Dict[str, Any]]:
        """Standard entry point for initial retrieval."""
        return self.initial_retrieval(session)

