"""
Unit Tests for Phase 2: Embeddings & Vector Retrieval
Validates:
1. EmbeddingEngine output dimension and L2 normalization.
2. VectorStore indexing and cosine search scoring.
3. Metadata filtering by doc_id.
4. get_chunk retrieval by chunk_id.
"""

import math
import os
import shutil
import tempfile
import pytest
from retrieval.embedder import EmbeddingEngine
from retrieval.vector_store import VectorStore, SearchResult


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingEngine.get_instance()


@pytest.fixture(scope="module")
def temp_store(embedder):
    temp_dir = tempfile.mkdtemp(prefix="campusiq_test_chroma_")
    store = VectorStore(
        persist_dir=temp_dir,
        collection_name="test_collection",
        embedder=embedder,
    )

    # Insert test data
    test_chunks = [
        {
            "chunk_id": "test_rules_p1_c0",
            "doc_id": "test_rules",
            "title": "Test Rules",
            "page": 1,
            "chunk_index": 0,
            "content": "Students must maintain minimum 75 percent attendance to sit in exams.",
            "token_count": 12,
            "applies_to": "Students enrolled from July 2019 onwards",
            "document_date": "2019-09-03",
        },
        {
            "chunk_id": "test_rules_p2_c0",
            "doc_id": "test_rules",
            "title": "Test Rules",
            "page": 2,
            "chunk_index": 0,
            "content": "A minimum CGPA of 5.0 is required for graduation from the BTech program.",
            "token_count": 14,
            "applies_to": "Students enrolled from July 2019 onwards",
            "document_date": "2019-09-03",
        },
        {
            "chunk_id": "test_hostel_p1_c0",
            "doc_id": "test_hostel",
            "title": "Test Hostel Rules",
            "page": 1,
            "chunk_index": 0,
            "content": "Hostel in-time curfew is strictly 10:30 PM for all undergraduate students.",
            "token_count": 13,
            "applies_to": "All UG students",
            "document_date": "2024-01-15",
        },
    ]

    import json
    chunks_path = os.path.join(temp_dir, "test_chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in test_chunks:
            f.write(json.dumps(c) + "\n")

    store.index_chunks(chunks_path)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedder_dimension_and_normalization(embedder):
    assert embedder.embedding_dim == 384, f"Expected 384 dimensions for bge-small, got {embedder.embedding_dim}"

    query_vec = embedder.embed_query("What is the attendance policy?")
    assert len(query_vec) == 384

    # Check L2 normalization: sqrt(sum(x^2)) ~= 1.0
    l2_norm = math.sqrt(sum(x * x for x in query_vec))
    assert abs(l2_norm - 1.0) < 1e-3, f"Query embedding not normalized, norm={l2_norm}"


def test_vector_store_search_ordering(temp_store):
    results = temp_store.search("attendance percentage requirement for exams", k=3)
    assert len(results) > 0

    top = results[0]
    assert top.doc_id == "test_rules"
    assert top.page == 1
    assert "75 percent attendance" in top.content
    assert 0.0 <= top.score <= 1.0
    assert top.applies_to == "Students enrolled from July 2019 onwards"
    assert top.document_date == "2019-09-03"
    assert "Date: 2019-09-03" in top.to_citation()
    assert "Applies to: Students enrolled from July 2019 onwards" in top.to_citation()
    assert "citation" in top.to_dict()

    # Scores should be in descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_store_metadata_filtering(temp_store):
    # Search with filter restricted to 'test_hostel'
    results = temp_store.search("curfew rules", k=3, filters={"doc_id": "test_hostel"})
    assert len(results) == 1
    assert results[0].doc_id == "test_hostel"
    assert results[0].page == 1
    assert results[0].applies_to == "All UG students"
    assert results[0].document_date == "2024-01-15"


def test_vector_store_get_chunk(temp_store):
    chunk = temp_store.get_chunk("test_rules_p2_c0")
    assert chunk is not None
    assert chunk.doc_id == "test_rules"
    assert chunk.page == 2
    assert chunk.applies_to == "Students enrolled from July 2019 onwards"
    assert chunk.document_date == "2019-09-03"
    assert "CGPA of 5.0" in chunk.content

    missing = temp_store.get_chunk("non_existent_chunk_id")
    assert missing is None
