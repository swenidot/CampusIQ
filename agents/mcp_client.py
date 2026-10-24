"""
CampusIQ MCP Client
Client wrapper for the authenticated read-only MCP server (Streamable HTTP transport).
Enforces:
- Hard cap of 5 tool calls per question session.
- Token loaded securely from environment (never logged or printed).
- Support for in-process TestClient or remote httpx Client or Mock client for testing.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

logger = logging.getLogger("campusiq.agents.mcp_client")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MAX_TOOL_CALLS = 5


class MCPClient:
    """
    Authenticated client for CampusIQ MCP Server.
    Provides search_documents and get_chunk tool execution.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        http_client: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ):
        self.token = token or os.getenv("MCP_AUTH_TOKEN") or os.getenv("MCP_SERVER_TOKEN", "")
        self.base_url = base_url
        self.http_client = http_client
        self.vector_store = vector_store
        self.max_tool_calls = max_tool_calls
        self.call_count = 0
        self._test_client_context = None

    def reset_session(self) -> None:
        """Resets the tool call count for a new question session."""
        self.call_count = 0

    def _get_client(self):
        if self.http_client is not None:
            return self.http_client

        from mcp_server.server import create_mcp_app
        from starlette.testclient import TestClient

        app = create_mcp_app(vector_store=self.vector_store, auth_token=self.token)
        self._test_client_context = TestClient(app)
        self.http_client = self._test_client_context.__enter__()
        return self.http_client

    def close(self):
        if self._test_client_context is not None:
            try:
                self._test_client_context.__exit__(None, None, None)
            except Exception:
                pass
            self._test_client_context = None
            self.http_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.call_count >= self.max_tool_calls:
            logger.warning(f"MCPClient hard cap reached: {self.call_count}/{self.max_tool_calls} tool calls.")
            return []

        self.call_count += 1
        client = self._get_client()

        headers = {
            "Authorization": f"Bearer {self.token}",
            "accept": "application/json, text/event-stream",
        }
        payload = {
            "jsonrpc": "2.0",
            "id": self.call_count,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        try:
            res = client.post("/mcp", json=payload, headers=headers)
            if res.status_code != 200:
                logger.error(f"MCP tool call {name} failed with status {res.status_code}")
                return []

            text = res.text
            # Parse Streamable HTTP SSE or direct JSON response
            json_body = None
            for line in text.splitlines():
                if line.startswith("data:"):
                    json_body = json.loads(line[5:].strip())
                    break

            if json_body is None:
                json_body = res.json()

            if "error" in json_body:
                logger.error(f"MCP tool call {name} returned error: {json_body['error']}")
                return []

            result = json_body.get("result", {})
            content = result.get("content", [])
            if not content:
                return []

            items = []
            for item in content:
                text_content = item.get("text", "")
                if text_content:
                    try:
                        data = json.loads(text_content)
                        if isinstance(data, list):
                            items.extend(data)
                        elif isinstance(data, dict):
                            items.append(data)
                    except Exception:
                        pass
            return items
        except Exception as e:
            logger.error(f"MCP tool call {name} exception: {e}")
            return []

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Calls MCP search_documents tool."""
        # Query length cap matching MCP server validation (<= 500 chars)
        q = query.strip()[:500]
        clamped_k = max(1, min(10, k))
        return self._call_tool("search_documents", {"query": q, "k": clamped_k})

    def get_chunk(
        self,
        doc_id: Optional[str] = None,
        page: Optional[int] = None,
        chunk_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Calls MCP get_chunk tool by doc_id & page or chunk_id string."""
        if chunk_id and (doc_id is None or page is None):
            import re
            m = re.match(r"^(.*)_p(\d+)(?:_c\d+)?$", chunk_id)
            if m:
                doc_id = m.group(1)
                page = int(m.group(2))
            else:
                return []
        if doc_id is None or page is None:
            return []
        return self._call_tool("get_chunk", {"doc_id": doc_id, "page": page})
