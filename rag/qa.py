"""
CampusIQ RAG QA Pipeline
Coordinates vector retrieval, two-stage abstention (similarity gate + LLM grounding),
prompt formatting with structured citations, and generation.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from retrieval.vector_store import VectorStore, SearchResult
from rag.prompt import SYSTEM_PROMPT, build_prompt
from rag.generator import GeminiGenerator

logger = logging.getLogger("campusiq.rag.qa")

ABSTENTION_ANSWER = "I could not find this in the documents."
DEFAULT_SIMILARITY_THRESHOLD = 0.65
DEFAULT_TOP_K = 5


class RAGPipeline:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        generator: Optional[GeminiGenerator] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.vector_store = vector_store or VectorStore()
        self.generator = generator or GeminiGenerator()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes grounded QA for a question:
        1. Dense retrieval of top-k chunks.
        2. Stage 1 Abstention: Score threshold gating.
        3. Stage 2 Abstention: Grounded generation with strict prompt constraint.
        Returns a structured dictionary response.
        """
        start_time = time.perf_counter()
        k = top_k if top_k is not None else self.top_k
        thresh = threshold if threshold is not None else self.similarity_threshold

        from observability.tracing import trace_span

        with trace_span("request", {
            "mode": "baseline",
            "model": self.generator.model,
        }) as req_span:
            retrieval_start = time.perf_counter()
            with trace_span("retrieval", {"k": k}):
                retrieved: List[SearchResult] = self.vector_store.search(question, k=k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

            top_score = retrieved[0].score if retrieved else 0.0

            # Stage 1: Similarity score threshold check
            if not retrieved or top_score < thresh:
                total_elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Abstaining on query '{question[:60]}...' - Top score {top_score:.4f} < threshold {thresh:.4f}"
                )
                req_span.set_attribute("abstained", True)
                req_span.set_attribute("abstention_stage", "stage_1_threshold")
                req_span.set_attribute("num_chunks", len(retrieved))
                req_span.set_attribute("tool_calls", 1)
                return {
                    "question": question,
                    "answer": ABSTENTION_ANSWER,
                    "abstained": True,
                    "abstention_stage": "stage_1_threshold",
                    "abstention_reason": "score_below_threshold",
                    "citations": [],
                    "retrieved": [c.to_dict() for c in retrieved],
                    "retrieved_chunks": [c.to_dict() for c in retrieved],
                    "top_score": top_score,
                    "model": None,
                    "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "latency_ms": round(total_elapsed_ms, 2),
                    "retrieval_latency_ms": round(retrieval_ms, 2),
                    "generation_latency_ms": 0.0,
                }

            # Stage 2: Grounded LLM Generation
            prompt = build_prompt(question, retrieved)
            gen_start = time.perf_counter()

            with trace_span("generation", {"model": self.generator.model, "num_chunks": len(retrieved)}):
                gen_result = self.generator.generate(
                    prompt=prompt,
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                )
            gen_ms = (time.perf_counter() - gen_start) * 1000
            raw_answer = gen_result["text"].strip()

            # Check for model abstention
            is_model_abstention = (
                raw_answer == ABSTENTION_ANSWER
                or raw_answer.lower().startswith("i could not find this in the documents")
                or raw_answer.lower().startswith("i cannot find this in the documents")
            )

            total_elapsed_ms = (time.perf_counter() - start_time) * 1000

            from rag.citations import extract_citations

            if is_model_abstention:
                final_answer = ABSTENTION_ANSWER
                final_citations = []
                abstained = True
                stage = "stage_2_model"
                reason = "model_abstention"
            else:
                final_answer = raw_answer
                final_citations = extract_citations(raw_answer, retrieved)
                abstained = False
                stage = None
                reason = None

            token_dict = {
                "input_tokens": gen_result.get("input_tokens", 0),
                "output_tokens": gen_result.get("output_tokens", 0),
                "total_tokens": gen_result.get("total_tokens", 0),
            }

            req_span.set_attribute("abstained", abstained)
            req_span.set_attribute("abstention_stage", stage or "")
            req_span.set_attribute("input_tokens", token_dict["input_tokens"])
            req_span.set_attribute("output_tokens", token_dict["output_tokens"])
            req_span.set_attribute("num_chunks", len(retrieved))
            req_span.set_attribute("tool_calls", 1)

            return {
                "question": question,
                "answer": final_answer,
                "abstained": abstained,
                "abstention_stage": stage,
                "abstention_reason": reason,
                "citations": final_citations,
                "retrieved": [c.to_dict() for c in retrieved],
                "retrieved_chunks": [c.to_dict() for c in retrieved],
                "top_score": top_score,
                "model": gen_result.get("model", self.generator.model),
                "tokens": token_dict,
            "token_usage": token_dict,
            "latency_ms": round(total_elapsed_ms, 2),
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "generation_latency_ms": round(gen_ms, 2),
            "cached": gen_result.get("cached", False),
        }
