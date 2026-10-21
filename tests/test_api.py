"""
Unit tests for CampusIQ FastAPI API (GET /health, POST /ask).
Uses FastAPI TestClient and mock pipeline/fake LLM with zero real API calls.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app, set_pipeline
from retrieval.vector_store import SearchResult
from rag.qa import RAGPipeline


class FakeLLMClient:
    def __init__(self, response_text: str = "Attendance must be 75%. [Academic Regulations, p.7]"):
        self.response_text = response_text
        self.call_count = 0

        class MockModels:
            def __init__(self, outer):
                self.outer = outer

            def generate_content(self, model, contents, config=None):
                self.outer.call_count += 1

                class MockResponse:
                    def __init__(self, text):
                        self.text = text
                        self.usage_metadata = type(
                            "Usage",
                            (),
                            {
                                "prompt_token_count": 50,
                                "candidates_token_count": 25,
                                "total_token_count": 75,
                            },
                        )()

                return MockResponse(self.outer.response_text)

        self.models = MockModels(self)


class FakeVectorStore:
    def __init__(self, results=None):
        self._results = results or []

    def search(self, query: str, k: int = 5):
        return self._results[:k]


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify GET /health returns 200 OK and expected service metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "CampusIQ" in data["service"]
    assert "version" in data


def test_ask_endpoint_success(client):
    """Verify POST /ask returns structured response with citations, retrieved, model, tokens, and latency."""
    chunk = SearchResult(
        chunk_id="chunk_test_1",
        content="Undergraduate attendance must be at least 75%.",
        score=0.88,
        distance=0.12,
        doc_id="rules_ug_2019",
        title="Academic Regulations and Degree Requirements",
        page=7,
        metadata={},
        file_date="2019-09-03",
        applies_to="July 2019 onwards",
    )
    fake_store = FakeVectorStore([chunk])
    answer_text = "Minimum attendance required is 75% [Academic Regulations and Degree Requirements, p.7]."
    fake_client = FakeLLMClient(answer_text)

    from rag.generator import GeminiGenerator
    fake_gen = GeminiGenerator(api_key="fake-key", client=fake_client, use_cache=False, model="gemini-3.5-flash-lite")
    mock_pipeline = RAGPipeline(vector_store=fake_store, generator=fake_gen, similarity_threshold=0.65)
    set_pipeline(mock_pipeline)

    payload = {"question": "What is the minimum attendance requirement?", "top_k": 3}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == answer_text
    assert data["abstained"] is False
    assert data["abstention_stage"] is None
    assert data["model"] == "gemini-3.5-flash-lite"
    assert isinstance(data["latency_ms"], (int, float))
    assert data["tokens"]["total_tokens"] == 75

    # Check citations and retrieved separation
    assert len(data["citations"]) == 1
    assert data["citations"][0]["doc_title"] == "Academic Regulations and Degree Requirements"
    assert data["citations"][0]["page"] == 7
    assert data["citations"][0]["invalid_citation"] is False

    assert len(data["retrieved"]) == 1
    assert data["retrieved"][0]["chunk_id"] == "chunk_test_1"

    # Reset pipeline
    set_pipeline(None)


def test_ask_endpoint_threshold_abstention(client):
    """Verify POST /ask returns Stage 1 abstention when score < threshold with no LLM invocation."""
    low_chunk = SearchResult(
        chunk_id="low_chunk_1",
        content="Irrelevant text.",
        score=0.45,
        distance=0.55,
        doc_id="rules_ug_2019",
        title="Academic Regulations",
        page=1,
        metadata={},
    )
    fake_store = FakeVectorStore([low_chunk])
    fake_client = FakeLLMClient("Should never be called")

    from rag.generator import GeminiGenerator
    fake_gen = GeminiGenerator(api_key="fake-key", client=fake_client, use_cache=False)
    mock_pipeline = RAGPipeline(vector_store=fake_store, generator=fake_gen, similarity_threshold=0.65)
    set_pipeline(mock_pipeline)

    payload = {"question": "What is the fee for swimming pool pass?"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I could not find this in the documents."
    assert data["abstained"] is True
    assert data["abstention_stage"] == "stage_1_threshold"
    assert data["model"] is None
    assert data["citations"] == []
    assert len(data["retrieved"]) == 1
    assert fake_client.call_count == 0

    # Reset pipeline
    set_pipeline(None)


def test_ask_endpoint_empty_question(client):
    """Verify POST /ask rejects empty questions with 400 Bad Request."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]
