"""
CampusIQ FastAPI Application
Provides REST endpoints:
- GET /health
- POST /ask
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.qa import RAGPipeline

app = FastAPI(
    title="CampusIQ API",
    description="CampusIQ Academic RAG Assistant for IIT Jodhpur",
    version="1.0.0",
)

# Pipeline holder (allows mock injection in tests)
pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = RAGPipeline()
    return pipeline_instance


def set_pipeline(pipeline: Optional[RAGPipeline]) -> None:
    global pipeline_instance
    pipeline_instance = pipeline


class AskRequest(BaseModel):
    question: str = Field(..., description="The query to answer based on IIT Jodhpur documents")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve (default: 5)")
    threshold: Optional[float] = Field(None, description="Similarity threshold for Stage 1 abstention (default: 0.65)")


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]
    abstained: bool
    abstention_stage: Optional[str] = None
    model: Optional[str] = None
    latency_ms: float
    tokens: Dict[str, int]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "CampusIQ RAG API", "version": "1.0.0"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    res = pipeline.answer(
        question=request.question,
        top_k=request.top_k,
        threshold=request.threshold,
    )

    return AskResponse(
        answer=res["answer"],
        citations=res.get("citations", []),
        retrieved=res.get("retrieved", []),
        abstained=res.get("abstained", False),
        abstention_stage=res.get("abstention_stage"),
        model=res.get("model"),
        latency_ms=res.get("latency_ms", 0.0),
        tokens=res.get("tokens", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    )
