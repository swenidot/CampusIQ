"""
CampusIQ FastAPI Application
Provides REST endpoints:
- GET /health
- POST /ask (supports mode="baseline" and mode="multi_agent")
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.qa import RAGPipeline
from agents.coordinator import MultiAgentPipeline

app = FastAPI(
    title="CampusIQ API",
    description="CampusIQ Academic RAG Assistant for IIT Jodhpur",
    version="1.0.0",
)

# Pipeline holders (allows mock injection in tests)
pipeline_instance: Optional[RAGPipeline] = None
multi_agent_pipeline_instance: Optional[MultiAgentPipeline] = None


def get_pipeline() -> RAGPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = RAGPipeline()
    return pipeline_instance


def set_pipeline(pipeline: Optional[RAGPipeline]) -> None:
    global pipeline_instance
    pipeline_instance = pipeline


def get_multi_agent_pipeline() -> MultiAgentPipeline:
    global multi_agent_pipeline_instance
    if multi_agent_pipeline_instance is None:
        multi_agent_pipeline_instance = MultiAgentPipeline()
    return multi_agent_pipeline_instance


def set_multi_agent_pipeline(pipeline: Optional[MultiAgentPipeline]) -> None:
    global multi_agent_pipeline_instance
    multi_agent_pipeline_instance = pipeline


class AskRequest(BaseModel):
    question: str = Field(..., description="The query to answer based on IIT Jodhpur documents")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve (default: 5)")
    threshold: Optional[float] = Field(None, description="Similarity threshold for Stage 1 abstention (default: 0.65)")
    mode: str = Field("baseline", description="Pipeline mode: 'baseline' or 'multi_agent'")


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]
    abstained: bool
    abstention_stage: Optional[str] = None
    model: Optional[str] = None
    latency_ms: float
    tokens: Dict[str, int]
    llm_calls: Optional[int] = None
    tool_calls: Optional[int] = None
    mode: Optional[str] = None


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "CampusIQ RAG API", "version": "1.0.0"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    selected_mode = request.mode.lower().strip()
    if selected_mode not in ["baseline", "multi_agent"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Allowed modes are 'baseline' and 'multi_agent'.",
        )

    if selected_mode == "multi_agent":
        ma_pipeline = get_multi_agent_pipeline()
        res = ma_pipeline.answer(question=request.question)
        return AskResponse(
            answer=res["answer"],
            citations=res.get("citations", []),
            retrieved=res.get("retrieved", []),
            abstained=res.get("abstained", False),
            abstention_stage=res.get("abstention_stage"),
            model=res.get("model"),
            latency_ms=res.get("latency_ms", 0.0),
            tokens=res.get("tokens", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
            llm_calls=res.get("llm_calls"),
            tool_calls=res.get("tool_calls"),
            mode="multi_agent",
        )

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
        mode="baseline",
    )
