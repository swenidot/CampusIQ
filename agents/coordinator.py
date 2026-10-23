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

    def answer(
        self,
        question: str,
        use_cache: Optional[bool] = None,
        pacing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes the multi-agent pipeline for a single question.
        """
        start_time = time.perf_counter()
        cache_enabled = self.use_cache if use_cache is None else use_cache

        cache_key = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
        if cache_enabled and cache_key in self.cache:
            cached_res = dict(self.cache[cache_key])
            cached_res["cached"] = True
            logger.info(f"Loaded multi-agent cached response for '{question[:40]}'")
            return cached_res

        self.mcp_client.reset_session()
        session = SessionState(question=question, model=self.model)

        # 1. Retriever Agent
        self.retriever.run(session)

        # 2. Answerer Agent
        self.answerer.run(session)

        # 3. Verifier Agent
        final_answer = self.verifier.run(session)

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000

        top_score = 0.0
        if session.chunks_seen:
            top_score = max((c.get("score", 0.0) for c in session.chunks_seen), default=0.0)

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
            "latency_ms": round(total_elapsed_ms, 2),
            "llm_calls": session.llm_call_count,
            "tool_calls": session.tool_call_count,
            "cached": False,
            "session_state": {
                "queries_tried": session.queries_tried,
                "chunks_count": len(session.chunks_seen),
                "verifier_result": session.verifier_result,
            },
        }

        if cache_enabled:
            self.cache[cache_key] = result
            self._save_cache()

        should_pace = pacing if pacing is not None else True
        if should_pace and self.pacing_seconds > 0 and session.llm_call_count > 0:
            time.sleep(self.pacing_seconds)

        return result
