"""
CampusIQ Authenticated Read-Only MCP Server
Transport: Streamable HTTP (Stateless)
Tools:
  - search_documents(query, k): Semantic search over verified IIT Jodhpur documents.
  - get_chunk(doc_id, page): Exact retrieval of chunks for a document page.

Security:
  - Bearer token authentication via MCP_AUTH_TOKEN environment variable.
  - Constant-time token verification (secrets.compare_digest).
  - Strict input validation (k: 1..10, query: <= 500 chars).
  - Strictly read-only; no write or filesystem manipulation endpoints.
  - Never logs secret token values.
"""

import logging
import os
import secrets
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http import TransportSecuritySettings
from retrieval.vector_store import VectorStore

logger = logging.getLogger("campusiq.mcp_server")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MAX_QUERY_LENGTH = 500
MIN_K = 1
MAX_K = 10


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates Authorization: Bearer <token> on all incoming MCP endpoints.
    Uses constant-time comparison. Never logs the token value.
    """

    def __init__(self, app, auth_token: Optional[str] = None):
        super().__init__(app)
        self.auth_token = auth_token or os.getenv("MCP_AUTH_TOKEN") or os.getenv("MCP_SERVER_TOKEN", "")

    async def dispatch(self, request: Request, call_next):
        # Allow /health endpoint without authentication
        if request.url.path in ["/health", "/healthz"]:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            logger.warning("MCP request rejected: Missing Authorization header")
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Unauthorized: Missing Bearer token"}},
                status_code=401,
            )

        if not auth_header.startswith("Bearer "):
            logger.warning("MCP request rejected: Malformed Authorization header")
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Unauthorized: Expected 'Bearer <token>'"}},
                status_code=401,
            )

        provided_token = auth_header[7:].strip()
        expected = self.auth_token or os.getenv("MCP_AUTH_TOKEN") or os.getenv("MCP_SERVER_TOKEN", "")

        if not expected:
            logger.error("MCP authentication error: No expected MCP_AUTH_TOKEN configured on server")
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Server configuration error"}},
                status_code=500,
            )

        if not secrets.compare_digest(provided_token, expected):
            logger.warning("MCP request rejected: Invalid bearer token")
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Unauthorized: Invalid token"}},
                status_code=401,
            )

        return await call_next(request)


def get_mcp_server(vector_store: Optional[VectorStore] = None) -> MCPServer:
    """Instantiates and registers read-only tools on MCPServer."""
    server = MCPServer(
        name="campusiq-mcp",
        version="1.0.0",
        description="CampusIQ Institutional Knowledge Search for IIT Jodhpur Academic Regulations and Governance.",
    )

    store = vector_store or VectorStore()

    @server.tool(
        name="search_documents",
        description="Performs semantic cosine search over indexed IIT Jodhpur regulatory documents. Returns top-k chunks with scores, dates, and applicability metadata. Query limited to 500 characters, k limited to 1-10.",
    )
    def search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Input validation
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"query length ({len(query)}) exceeds maximum allowed ({MAX_QUERY_LENGTH} characters).")
        if not isinstance(k, int) or k < MIN_K or k > MAX_K:
            raise ValueError(f"k must be an integer between {MIN_K} and {MAX_K}.")

        results = store.search(query=query.strip(), k=k)
        return [
            {
                "doc_id": r.doc_id,
                "title": r.title,
                "page": r.page,
                "applies_to": r.applies_to,
                "file_date": r.file_date,
                "score": r.score,
                "text": r.content,
            }
            for r in results
        ]

    @server.tool(
        name="get_chunk",
        description="Retrieves all chunks belonging to a specific document ID and printed page number. Strictly read-only.",
    )
    def get_chunk(doc_id: str, page: int) -> List[Dict[str, Any]]:
        # Input validation
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")
        if any(bad in doc_id for bad in ["..", "/", "\\"]):
            raise ValueError("Invalid doc_id: Path traversal characters are forbidden.")
        if not isinstance(page, int) or page < 1:
            raise ValueError("page must be an integer >= 1.")

        results = store.get_page_chunks(doc_id=doc_id.strip(), page=page)
        return [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "page": r.page,
                "applies_to": r.applies_to,
                "file_date": r.file_date,
                "score": r.score,
                "text": r.content,
            }
            for r in results
        ]

    return server


async def health_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "CampusIQ MCP Server",
        "transport": "streamable_http",
        "version": "1.0.0",
        "read_only": True,
    })


def create_mcp_app(
    vector_store: Optional[VectorStore] = None,
    auth_token: Optional[str] = None,
    enable_dns_rebinding_protection: bool = False,
) -> Starlette:
    """
    Creates the complete authenticated Starlette ASGI application
    using MCP 2.x Streamable HTTP transport.
    """
    server = get_mcp_server(vector_store=vector_store)

    transport_sec = TransportSecuritySettings(
        enable_dns_rebinding_protection=enable_dns_rebinding_protection,
        allowed_hosts=["localhost:*", "127.0.0.1:*", "testserver"],
    )

    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        transport_security=transport_sec,
    )

    # Add health route
    app.routes.append(Route("/health", health_endpoint, methods=["GET"]))

    # Wrap with Bearer Authentication Middleware
    app.add_middleware(BearerAuthMiddleware, auth_token=auth_token)

    return app


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", "8001"))

    print(f"Starting CampusIQ Authenticated MCP Server on http://{host}:{port}/mcp (Streamable HTTP)")
    asgi_app = create_mcp_app()
    uvicorn.run(asgi_app, host=host, port=port, log_level="info")
