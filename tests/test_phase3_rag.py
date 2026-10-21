"""
Unit tests for Phase 3: RAG Pipeline, Citations, Abstention, and Security.
Validates:
1. Citation formatting includes title, page, date, and applies_to.
2. Abstention path triggers on score below threshold without calling LLM.
3. Abstention path triggers when LLM signals lack of evidence.
4. Successful grounded answer includes citations and proper metadata.
5. Zero secrets or API keys are hardcoded in source code files.
"""

import os
import re
import pytest
from retrieval.vector_store import SearchResult
from rag.qa import RAGPipeline, ABSTENTION_ANSWER


class FakeLLMClient:
    """Mock Gemini client that simulates LLM generation without network requests."""
    def __init__(self, response_text: str = "According to the regulations, attendance must be 75%. [Academic Regulations, p.7]"):
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
                                "prompt_token_count": 45,
                                "candidates_token_count": 20,
                                "total_token_count": 65,
                            },
                        )()

                return MockResponse(self.outer.response_text)

        self.models = MockModels(self)


class FakeVectorStore:
    """Mock VectorStore returning controlled SearchResult chunks."""
    def __init__(self, results=None):
        self._results = results or []

    def search(self, query: str, k: int = 5):
        return self._results[:k]


def test_citation_formatting():
    """Verify SearchResult.to_citation formats title, page, file_date, and applies_to correctly."""
    result = SearchResult(
        chunk_id="chunk_1",
        content="Sample regulations text",
        score=0.85,
        distance=0.15,
        doc_id="rules_ug_2019",
        title="Academic Regulations and Degree Requirements",
        page=7,
        metadata={"category": "Academic Rules"},
        applies_to="Students enrolled from July 2019 onwards",
        file_date="2019-09-03",
    )

    citation = result.to_citation()
    assert "[Academic Regulations and Degree Requirements, p.7]" in citation
    assert "Date: 2019-09-03" in citation
    assert "Applies to: Students enrolled from July 2019 onwards" in citation


def test_abstention_score_threshold():
    """Verify that when top retrieval score < threshold, pipeline abstains without LLM invocation."""
    low_score_chunk = SearchResult(
        chunk_id="low_1",
        content="Irrelevant text",
        score=0.45,  # Below threshold 0.65
        distance=0.55,
        doc_id="rules_ug_2019",
        title="Academic Regulations",
        page=1,
        metadata={},
    )
    fake_store = FakeVectorStore([low_score_chunk])
    fake_client = FakeLLMClient("Should not be returned")

    from rag.generator import GeminiGenerator
    fake_gen = GeminiGenerator(
        api_key="fake-key",
        client=fake_client,
        use_cache=False,
    )

    pipeline = RAGPipeline(
        vector_store=fake_store,
        generator=fake_gen,
        similarity_threshold=0.65,
    )

    result = pipeline.answer("What is the pool refund policy?")
    assert result["answer"] == ABSTENTION_ANSWER
    assert result["abstained"] is True
    assert result["abstention_reason"] == "score_below_threshold"
    assert result["citations"] == []
    assert fake_client.call_count == 0, "LLM must not be called when score is below threshold"


def test_abstention_llm_path():
    """Verify that when retrieved chunks lack answer, model outputs abstention and pipeline marks it."""
    high_score_chunk = SearchResult(
        chunk_id="high_1",
        content="Some regulations that do not answer the question.",
        score=0.75,  # Above threshold
        distance=0.25,
        doc_id="rules_ug_2019",
        title="Academic Regulations",
        page=2,
        metadata={},
    )
    fake_store = FakeVectorStore([high_score_chunk])
    fake_client = FakeLLMClient("I could not find this in the documents.")

    from rag.generator import GeminiGenerator
    fake_gen = GeminiGenerator(
        api_key="fake-key",
        client=fake_client,
        use_cache=False,
    )

    pipeline = RAGPipeline(
        vector_store=fake_store,
        generator=fake_gen,
        similarity_threshold=0.65,
    )

    result = pipeline.answer("What is the penalty for indiscipline?")
    assert result["answer"] == ABSTENTION_ANSWER
    assert result["abstained"] is True
    assert result["abstention_reason"] == "model_abstention"
    assert result["citations"] == []
    assert fake_client.call_count == 1


def test_successful_grounded_answer():
    """Verify grounded answer retains citations and metadata, and separates retrieved list."""
    good_chunk = SearchResult(
        chunk_id="good_1",
        content="Attendance must be at least 75%.",
        score=0.82,
        distance=0.18,
        doc_id="rules_ug_2019",
        title="Academic Regulations",
        page=7,
        metadata={},
        file_date="2019-09-03",
        applies_to="July 2019 onwards",
    )
    fake_store = FakeVectorStore([good_chunk])
    answer_text = "Minimum attendance required is 75% [Academic Regulations, p.7]."
    fake_client = FakeLLMClient(answer_text)

    from rag.generator import GeminiGenerator
    fake_gen = GeminiGenerator(
        api_key="fake-key",
        client=fake_client,
        use_cache=False,
    )

    pipeline = RAGPipeline(
        vector_store=fake_store,
        generator=fake_gen,
        similarity_threshold=0.65,
    )

    result = pipeline.answer("What is the minimum attendance requirement?")
    assert result["answer"] == answer_text
    assert result["abstained"] is False
    assert len(result["citations"]) == 1
    cit = result["citations"][0]
    assert cit["doc_title"] == "Academic Regulations"
    assert cit["page"] == 7
    assert cit["applies_to"] == "July 2019 onwards"
    assert cit["file_date"] == "2019-09-03"
    assert cit["invalid_citation"] is False
    assert len(result["retrieved"]) == 1
    assert result["retrieved"][0]["chunk_id"] == "good_1"
    assert result["tokens"]["total_tokens"] == 65


def test_citation_validation_and_invalid_flagging():
    """
    Verify requirement A.1:
    1. citations contains only sources cited in answer text.
    2. Maps to chunk applies_to and file_date.
    3. Any cited marker matching no retrieved chunk is flagged invalid_citation=True.
    4. retrieved contains all retrieved chunks.
    """
    chunk1 = SearchResult(
        chunk_id="rules_ug_p10",
        content="Grading system details",
        score=0.88,
        distance=0.12,
        doc_id="rules_ug_2019",
        title="Academic Regulations and Degree Requirements",
        page=10,
        metadata={},
        file_date="2019-09-03",
        applies_to="July 2019 onwards",
    )
    chunk2 = SearchResult(
        chunk_id="calendar_p1",
        content="Academic calendar details",
        score=0.78,
        distance=0.22,
        doc_id="academic_calendar_2026",
        title="Academic Calendar",
        page=1,
        metadata={},
        file_date="2026-07-30",
        applies_to="AY 2026-27",
    )

    fake_store = FakeVectorStore([chunk1, chunk2])

    # The model cites chunk1 correctly, but hallucinates a citation to a non-existent document/page
    simulated_answer = (
        "Letter grades are defined on a 10-point scale [Academic Regulations and Degree Requirements, p.10]. "
        "Also see non-existent guidelines [Fictional Handbook, p.99]."
    )
    fake_client = FakeLLMClient(simulated_answer)

    from rag.generator import GeminiGenerator
    fake_gen = GeminiGenerator(api_key="fake-key", client=fake_client, use_cache=False)
    pipeline = RAGPipeline(vector_store=fake_store, generator=fake_gen, similarity_threshold=0.65)

    result = pipeline.answer("Explain grading")
    assert result["abstained"] is False

    # retrieved must contain ALL retrieved chunks (both chunk1 and chunk2)
    assert len(result["retrieved"]) == 2
    assert result["retrieved"][0]["chunk_id"] == "rules_ug_p10"
    assert result["retrieved"][1]["chunk_id"] == "calendar_p1"

    # citations must contain only the 2 cited markers (chunk2 was retrieved but NOT cited in text)
    assert len(result["citations"]) == 2

    # First citation is valid and matched
    valid_cit = next(c for c in result["citations"] if "Academic Regulations" in c["doc_title"])
    assert valid_cit["page"] == 10
    assert valid_cit["applies_to"] == "July 2019 onwards"
    assert valid_cit["file_date"] == "2019-09-03"
    assert valid_cit["invalid_citation"] is False

    # Second citation is invalid and flagged
    invalid_cit = next(c for c in result["citations"] if "Fictional Handbook" in c["doc_title"])
    assert invalid_cit["page"] == 99
    assert invalid_cit["invalid_citation"] is True
    assert invalid_cit["chunk_id"] is None


def test_no_secrets_in_code():
    """Scan all codebase python files to ensure no API keys or secrets are committed or hardcoded."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dirs = ["rag", "retrieval", "ingestion", "data"]

    # Patterns matching common secret formats
    secret_patterns = [
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"AQ\.Ab8[0-9A-Za-z-_]{40,}"),
    ]

    for d in target_dirs:
        dir_path = os.path.join(base_dir, d)
        if not os.path.exists(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    for pat in secret_patterns:
                        match = pat.search(content)
                        assert not match, f"Potential secret found in {full_path}: {match.group(0)[:8]}..."
