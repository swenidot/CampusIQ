"""
Unit tests for Phase 7 OpenTelemetry Structured Tracing Module.
Validates:
1. Expected span names exist and nest properly under 'request' (root).
2. Token attributes (input_tokens, output_tokens) and question_id are recorded accurately.
3. Strict privacy guard: no prompt text, chunk text, keys, or bearer tokens in attributes.
"""

import pytest
from observability.tracing import (
    setup_tracing,
    get_tracer,
    trace_span,
    get_recent_spans,
    clear_recent_spans,
    sanitize_attributes,
    FORBIDDEN_KEY_SUBSTRINGS,
)
from agents.coordinator import MultiAgentPipeline
from agents.session import SessionState


class MockGenerator:
    def __init__(self):
        self.model = "gemini-3.5-flash-lite"

    def generate(self, prompt: str, system_instruction: str = "", temperature: float = 0.0):
        return {
            "text": 'Students must maintain "minimum attendance of 75%" [Academic Regulations, p.7].',
            "input_tokens": 125,
            "output_tokens": 30,
            "total_tokens": 155,
            "model": self.model,
        }


class MockMCPClient:
    def __init__(self):
        self.call_count = 0
        self.max_tool_calls = 5

    def reset_session(self):
        self.call_count = 0

    def search_documents(self, query: str, k: int = 5):
        self.call_count += 1
        return [
            {
                "chunk_id": "rules_ug_2019_7",
                "doc_id": "rules_ug_2019",
                "title": "Academic Regulations",
                "page": 7,
                "score": 0.88,
                "text": "A student must maintain a minimum attendance of 75% in all lectures.",
            }
        ]

    def get_chunk(self, doc_id: str, page: int):
        self.call_count += 1
        return [
            {
                "chunk_id": f"{doc_id}_{page}",
                "doc_id": doc_id,
                "title": "Academic Regulations",
                "page": page,
                "score": 0.80,
                "text": f"Continuation content on page {page}.",
            }
        ]


def test_span_nesting_and_structure(tmp_path):
    """Verify expected span names exist and nest properly under 'request' root span."""
    test_trace_file = str(tmp_path / "test_traces.jsonl")
    setup_tracing(file_path=test_trace_file, force_reinit=True)
    clear_recent_spans()

    fake_mcp = MockMCPClient()
    fake_gen = MockGenerator()
    pipeline = MultiAgentPipeline(
        mcp_client=fake_mcp,
        generator=fake_gen,
        use_cache=False,
        pacing_seconds=0.0,
    )

    res = pipeline.answer("What is the minimum attendance requirement?", question_id="q01")
    assert "answer" in res

    spans = get_recent_spans()
    span_names = [s["name"] for s in spans]

    # Verify required spans are present
    assert "request" in span_names
    assert "retrieval" in span_names
    assert "generation" in span_names
    assert "verification" in span_names

    # Verify hierarchy: child spans must have parent_id == root request span_id
    root_span = next(s for s in spans if s["name"] == "request")
    root_id = root_span["context"]["span_id"]

    child_spans = [s for s in spans if s["name"] in ["retrieval", "generation", "verification"]]
    assert len(child_spans) >= 3
    for child in child_spans:
        assert child["parent_id"] == root_id, f"Span {child['name']} not nested under root request"


def test_token_and_question_id_attributes(tmp_path):
    """Verify token attributes and question_id are recorded accurately."""
    test_trace_file = str(tmp_path / "test_traces.jsonl")
    setup_tracing(file_path=test_trace_file, force_reinit=True)
    clear_recent_spans()

    fake_mcp = MockMCPClient()
    fake_gen = MockGenerator()
    pipeline = MultiAgentPipeline(
        mcp_client=fake_mcp,
        generator=fake_gen,
        use_cache=False,
        pacing_seconds=0.0,
    )

    pipeline.answer("What is the minimum attendance requirement?", question_id="q07")

    spans = get_recent_spans()
    root_span = next(s for s in spans if s["name"] == "request")
    attrs = root_span["attributes"]

    assert attrs.get("question_id") == "q07"
    assert attrs.get("mode") == "multi_agent"
    assert attrs.get("model") == "gemini-3.5-flash-lite"
    assert attrs.get("input_tokens", 0) > 0
    assert attrs.get("output_tokens", 0) > 0
    assert "tool_calls" in attrs


def test_privacy_guard_no_sensitive_data(tmp_path):
    """Verify strictly no prompt text, chunk text, API keys, or bearer tokens in span attributes."""
    # Test sanitizer directly
    dirty_attributes = {
        "question_id": "q12",
        "prompt_text": "Sensitive user prompt about exams",
        "chunk_text": "Full document content with confidential rules",
        "api_key": "AIzaSyD-secret-key-12345",
        "bearer_token": "Bearer eyJhbGciOi...",
        "input_tokens": 120,
        "output_tokens": 45,
        "model": "gemini-3.5-flash-lite",
        "raw_text": "A very long text snippet that shouldn't be included\nwith newlines",
    }

    clean = sanitize_attributes(dirty_attributes)

    assert "question_id" in clean
    assert "input_tokens" in clean
    assert "output_tokens" in clean
    assert "model" in clean

    # Forbidden fields MUST be stripped
    assert "prompt_text" not in clean
    assert "chunk_text" not in clean
    assert "api_key" not in clean
    assert "bearer_token" not in clean
    assert "raw_text" not in clean

    # Test attributes on emitted spans across full execution
    test_trace_file = str(tmp_path / "test_traces.jsonl")
    setup_tracing(file_path=test_trace_file, force_reinit=True)
    clear_recent_spans()

    fake_mcp = MockMCPClient()
    fake_gen = MockGenerator()
    pipeline = MultiAgentPipeline(mcp_client=fake_mcp, generator=fake_gen, use_cache=False, pacing_seconds=0.0)
    pipeline.answer("Test question?", question_id="q99")

    spans = get_recent_spans()
    for span in spans:
        for k, v in span["attributes"].items():
            k_lower = k.lower()
            assert not any(bad in k_lower for bad in FORBIDDEN_KEY_SUBSTRINGS), f"Forbidden key in span: {k}"
            if isinstance(v, str):
                assert not v.startswith("Bearer "), f"Bearer token found in span value: {v}"
                assert "AIzaSy" not in v, f"API key pattern found in span value: {v}"
                assert "\n" not in v, f"Multi-line text found in span value: {v}"
