"""
CampusIQ Multi-Agent Session State
Simple explicit session state tracking without hidden memory.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    """
    Explicit, inspectable state for a multi-agent QA session.
    Tracks queries tried, chunks seen, tool calls, and verifier decisions.
    """
    question: str
    queries_tried: List[str] = field(default_factory=list)
    chunks_seen: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    llm_call_count: int = 0
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None
    verifier_result: Optional[Dict[str, Any]] = None
    abstained: bool = False
    abstention_reason: Optional[str] = None
    abstention_stage: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    model: str = "gemini-3.5-flash-lite"
    tokens: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    verification_path: Optional[str] = None  # "exact_normalized", "fuzzy", or "failed"
    verification_paths: List[str] = field(default_factory=list)
    compute_ms: float = 0.0
    wall_clock_ms: float = 0.0
    top_score: float = 0.0

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Adds a chunk if not already present in chunks_seen."""
        if not chunk.get("chunk_id") and chunk.get("doc_id") and chunk.get("page"):
            chunk["chunk_id"] = f"{chunk.get('doc_id')}_{chunk.get('page')}"
        text_snip = (chunk.get("text") or chunk.get("content") or "")[:40]
        cid = chunk.get("chunk_id") or f"{chunk.get('doc_id')}_{chunk.get('page')}_{text_snip}"
        existing_ids = {
            c.get("chunk_id") or f"{c.get('doc_id')}_{c.get('page')}_{(c.get('text') or c.get('content') or '')[:40]}"
            for c in self.chunks_seen
        }
        if cid not in existing_ids:
            self.chunks_seen.append(chunk)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        for c in chunks:
            self.add_chunk(c)

    def add_tokens(self, in_tok: int, out_tok: int) -> None:
        self.tokens["input_tokens"] += in_tok
        self.tokens["output_tokens"] += out_tok
        self.tokens["total_tokens"] += (in_tok + out_tok)
