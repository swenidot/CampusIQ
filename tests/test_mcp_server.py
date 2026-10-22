"""
Unit tests for CampusIQ Authenticated Read-Only MCP Server.
Validates:
1. Valid bearer token works (200 OK).
2. Missing and wrong tokens are rejected (401 Unauthorized).
3. Input validation rejects bad k (k < 1, k > 10) and overlong queries (> 500 chars).
4. Tools are strictly read-only (only search_documents and get_chunk).
5. No real network or LLM calls are made.
"""

import pytest
from starlette.testclient import TestClient
from mcp_server.server import create_mcp_app, MAX_QUERY_LENGTH
from retrieval.vector_store import SearchResult

TEST_TOKEN = "test-secret-bearer-token-xyz123"


class FakeVectorStore:
    def __init__(self):
        self.search_called = False
        self.get_page_called = False

    def search(self, query: str, k: int = 5):
        self.search_called = True
        return [
            SearchResult(
                chunk_id="chunk_1",
                content="Mock content for search",
                score=0.91,
                distance=0.09,
                doc_id="rules_ug_2019",
                title="Academic Regulations",
                page=7,
                metadata={},
                applies_to="July 2019 onwards",
                file_date="2019-09-03",
            )
        ]

    def get_page_chunks(self, doc_id: str, page: int):
        self.get_page_called = True
        return [
            SearchResult(
                chunk_id="chunk_p7_1",
                content="Mock content for page 7",
                score=1.0,
                distance=0.0,
                doc_id=doc_id,
                title="Academic Regulations",
                page=page,
                metadata={},
                applies_to="July 2019 onwards",
                file_date="2019-09-03",
            )
        ]


@pytest.fixture
def mock_store():
    return FakeVectorStore()


@pytest.fixture
def app(mock_store):
    return create_mcp_app(
        vector_store=mock_store,
        auth_token=TEST_TOKEN,
        enable_dns_rebinding_protection=False,
    )


def test_mcp_health_endpoint(app):
    """Verify /health returns 200 without authentication."""
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["read_only"] is True


def test_mcp_unauthenticated_rejected(app):
    """Verify requests lacking Authorization header receive 401 Unauthorized."""
    with TestClient(app) as client:
        res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert res.status_code == 401
        data = res.json()
        assert "error" in data
        assert "Unauthorized" in data["error"]["message"]


def test_mcp_wrong_token_rejected(app):
    """Verify requests with invalid bearer token receive 401 Unauthorized."""
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer wrong-token-value"}
        res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}, headers=headers)
        assert res.status_code == 401
        data = res.json()
        assert "error" in data


def test_mcp_authenticated_tools_list(app):
    """Verify valid token succeeds and exposes only read-only tools."""
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}", "accept": "application/json, text/event-stream"}
        res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}, headers=headers)
        assert res.status_code == 200
        text = res.text
        # Check that only read-only tools are exposed
        assert "search_documents" in text
        assert "get_chunk" in text
        assert "write" not in text
        assert "delete" not in text
        assert "update" not in text


def test_mcp_search_documents_success(app, mock_store):
    """Verify search_documents tool returns structured chunks with score, text, and metadata."""
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}", "accept": "application/json, text/event-stream"}
        res = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_documents", "arguments": {"query": "attendance requirement", "k": 3}},
            },
            headers=headers,
        )
        assert res.status_code == 200
        assert mock_store.search_called is True
        assert "rules_ug_2019" in res.text
        assert "Academic Regulations" in res.text


def test_mcp_search_documents_validation(app):
    """Verify input validation rejects overlong query and out-of-range k."""
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}", "accept": "application/json, text/event-stream"}

        # Overlong query (> 500 chars)
        long_query = "x" * 501
        res_long = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search_documents", "arguments": {"query": long_query, "k": 5}}},
            headers=headers,
        )
        assert res_long.status_code == 200
        assert '"isError":true' in res_long.text or '"isError": true' in res_long.text

        # Bad k: k > 10
        res_bad_k = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "search_documents", "arguments": {"query": "valid query", "k": 15}}},
            headers=headers,
        )
        assert res_bad_k.status_code == 200
        assert '"isError":true' in res_bad_k.text or '"isError": true' in res_bad_k.text

        # Bad k: k < 1
        res_zero_k = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "search_documents", "arguments": {"query": "valid query", "k": 0}}},
            headers=headers,
        )
        assert res_zero_k.status_code == 200
        assert '"isError":true' in res_zero_k.text or '"isError": true' in res_zero_k.text


def test_mcp_get_chunk_validation_and_success(app, mock_store):
    """Verify get_chunk input validation and successful execution."""
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}", "accept": "application/json, text/event-stream"}

        # Success case
        res = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "get_chunk", "arguments": {"doc_id": "rules_ug_2019", "page": 7}}},
            headers=headers,
        )
        assert res.status_code == 200
        assert mock_store.get_page_called is True
        assert "rules_ug_2019" in res.text
        assert '"isError":false' in res.text or '"isError": false' in res.text

        # Traversal rejection
        res_bad_doc = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "get_chunk", "arguments": {"doc_id": "../etc/passwd", "page": 1}}},
            headers=headers,
        )
        assert '"isError":true' in res_bad_doc.text or '"isError": true' in res_bad_doc.text

        # Negative page
        res_bad_p = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "get_chunk", "arguments": {"doc_id": "rules_ug_2019", "page": 0}}},
            headers=headers,
        )
        assert '"isError":true' in res_bad_p.text or '"isError": true' in res_bad_p.text
