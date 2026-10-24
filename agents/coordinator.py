"""
CampusIQ Multi-Agent Coordinator
Orchestrates Retriever, Answerer, and Verifier agents with explicit SessionState.
Integrates response caching and free-tier rate-limit pacing.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from agents.session import SessionState
from agents.mcp_client import MCPClient
from agents.retriever import RetrieverAgent
from agents.answerer import AnswererAgent
from agents.verifier import VerifierAgent
from rag.generator import GeminiGenerator

logger = logging.getLogger("campusiq.agents.coordinator")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(BASE_DIR, "eval", "results", "multi_agent_v1_cache.json")
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class MultiAgentPipeline:
    """
    Coordinates the multi-agent system:
    RetrieverAgent (calls MCP search_documents & get_chunk) ->
    AnswererAgent (generates claim-quote draft) ->
    VerifierAgent (deterministic quote check + entailment check)
    """

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        generator: Optional[GeminiGenerator] = None,
        model: str = DEFAULT_MODEL,
        use_cache: bool = True,
        cache_file: str = CACHE_FILE,
        pacing_seconds: float = 4.5,
    ):
        self.model = model
        self.mcp_client = mcp_client or MCPClient()
        self.generator = generator or GeminiGenerator(model=model, use_cache=False)
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.pacing_seconds = pacing_seconds

        # Instantiate agents
        self.retriever = RetrieverAgent(self.mcp_client, generator=self.generator, model=model)
        self.answerer = AnswererAgent(generator=self.generator, model=model)
        self.verifier = VerifierAgent(generator=self.generator, model=model)

        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load multi-agent cache: {e}")
        return {}

    def _save_cache(self) -> None:
        if self.use_cache:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save multi-agent cache: {e}")

    @staticmethod
    def _resolve_citation_chunk_ids(citations: List[Dict[str, Any]], retrieved: List[Dict[str, Any]]) -> None:
        """Resolves missing chunk_id on citations by matching against retrieved chunks."""
        if not citations or not retrieved:
            return
        ret_map = {}
        for r in retrieved:
            title = r.get("title", "")
            page = r.get("page")
            cid = r.get("chunk_id") or f"{r.get('doc_id')}_{page}"
            if title and page is not None and cid:
                ret_map[(title, page)] = cid
            if r.get("doc_id") and page is not None and cid:
                ret_map[(r.get("doc_id"), page)] = cid

        for c in citations:
            if not c.get("chunk_id"):
                t = c.get("doc_title", "")
                p = c.get("page")
                if (t, p) in ret_map:
                    c["chunk_id"] = ret_map[(t, p)]
                else:
                    for (rt, rp), rcid in ret_map.items():
                        if rp == p and (rt.lower() in t.lower() or t.lower() in rt.lower()):
                            c["chunk_id"] = rcid
                            break

    def answer(
        self,
        question: str,
        question_id: Optional[str] = None,
        use_cache: Optional[bool] = None,
        pacing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes the multi-agent pipeline for a single question with OpenTelemetry tracing.
        """
        start_time = time.perf_counter()
        cache_enabled = self.use_cache if use_cache is None else use_cache

        cache_key = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
        is_cache_hit = bool(cache_enabled and cache_key in self.cache)

        from observability.tracing import trace_span

        with trace_span("request", {
            "question_id": question_id or "",
            "mode": "multi_agent",
            "model": self.model,
            "cache_hit": is_cache_hit,
        }) as req_span:
            if is_cache_hit:
                cached_res = dict(self.cache[cache_key])
                cached_res["cached"] = True
                # Ensure citation chunk_ids are resolved natively
                self._resolve_citation_chunk_ids(cached_res.get("citations", []), cached_res.get("retrieved", []))
                req_span.set_attribute("abstained", cached_res.get("abstained", False))
                req_span.set_attribute("abstention_stage", cached_res.get("abstention_stage") or "")
                req_span.set_attribute("verification_path", cached_res.get("verification_path") or "")
                req_span.set_attribute("input_tokens", cached_res.get("tokens", {}).get("input_tokens", 0))
                req_span.set_attribute("output_tokens", cached_res.get("tokens", {}).get("output_tokens", 0))
                req_span.set_attribute("tool_calls", cached_res.get("tool_calls", 0))
                req_span.set_attribute("num_chunks", len(cached_res.get("retrieved", [])))
                logger.info(f"Loaded multi-agent cached response for '{question[:40]}'")
                return cached_res

            self.mcp_client.reset_session()
            session = SessionState(question=question, model=self.model)

            # 1. Retriever Agent (Baseline-first search + adjacent expansion)
            with trace_span("retrieval", {"tool_name": "search_documents", "k": 5}):
                self.retriever.run(session)

            # Fast-exit if below relevance threshold (unanswerables exit with 0 LLM calls)
            if session.abstained and session.abstention_stage == "stage_1_threshold":
                final_answer = session.final_answer or "I could not find this in the documents."
            else:
                # 2. Answerer Agent
                with trace_span("generation", {"model": self.model, "num_chunks": len(session.chunks_seen)}):
                    self.answerer.run(session)

                # 3. Verifier Agent
                with trace_span("verification", {"verification_path": session.verification_path or ""}):
                    final_answer = self.verifier.run(session)

                # 4. Targeted Retry (if answerer abstained or verifier rejected, and cap allows)
                if session.abstained and self.mcp_client.call_count < self.mcp_client.max_tool_calls:
                    logger.info(f"Targeted retry initiated for question: '{question[:40]}'")
                    session.abstained = False
                    session.abstention_stage = None
                    session.abstention_reason = None
                    session.draft_answer = None
                    session.final_answer = None
                    session.citations = []

                    with trace_span("retry", {"k": 8}):
                        # Run targeted retry with k=8
                        self.retriever.targeted_retry(session)

                        # Re-run answerer and verifier
                        with trace_span("generation", {"model": self.model, "is_retry": True}):
                            self.answerer.run(session)
                        with trace_span("verification", {"is_retry": True}):
                            final_answer = self.verifier.run(session)

            wall_clock_ms = (time.perf_counter() - start_time) * 1000
            compute_ms = session.compute_ms

            top_score = session.top_score

            # Build clean retrieved list
            retrieved_list = []
            for c in session.chunks_seen:
                retrieved_list.append({
                    "chunk_id": c.get("chunk_id") or f"{c.get('doc_id')}_{c.get('page')}",
                    "doc_id": c.get("doc_id"),
                    "title": c.get("title"),
                    "page": c.get("page"),
                    "applies_to": c.get("applies_to"),
                    "file_date": c.get("file_date") or c.get("document_date"),
                    "score": c.get("score", 0.0),
                    "content": c.get("text") or c.get("content", ""),
                })

            # Ensure live citation chunk_ids are resolved natively
            self._resolve_citation_chunk_ids(session.citations, retrieved_list)

            # Set root span attributes
            req_span.set_attribute("abstained", session.abstained)
            req_span.set_attribute("abstention_stage", session.abstention_stage or "")
            req_span.set_attribute("verification_path", session.verification_path or "")
            req_span.set_attribute("input_tokens", session.tokens.get("input_tokens", 0))
            req_span.set_attribute("output_tokens", session.tokens.get("output_tokens", 0))
            req_span.set_attribute("tool_calls", session.tool_call_count)
            req_span.set_attribute("num_chunks", len(session.chunks_seen))

        result = {
            "question": question,
            "answer": final_answer,
            "abstained": session.abstained,
            "abstention_stage": session.abstention_stage,
            "abstention_reason": session.abstention_reason,
            "citations": session.citations,
            "retrieved": retrieved_list,
            "top_score": round(top_score, 4),
            "model": self.model,
            "tokens": session.tokens,
            "token_usage": session.tokens,
            "latency_ms": round(wall_clock_ms, 2),
            "wall_clock_ms": round(wall_clock_ms, 2),
            "compute_ms": round(compute_ms, 2),
            "verification_path": session.verification_path,
            "verification_paths": session.verification_paths,
            "llm_calls": session.llm_call_count,
            "tool_calls": session.tool_call_count,
            "cached": False,
            "session_state": {
                "queries_tried": session.queries_tried,
                "chunks_count": len(session.chunks_seen),
                "verifier_result": session.verifier_result,
                "verification_path": session.verification_path,
            },
        }

        if cache_enabled:
            self.cache[cache_key] = result
            self._save_cache()

        should_pace = pacing if pacing is not None else True
        if should_pace and self.pacing_seconds > 0 and session.llm_call_count > 0:
            time.sleep(self.pacing_seconds)

        return result
