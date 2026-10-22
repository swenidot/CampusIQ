"""
CampusIQ MCP Server Package
Authenticated, read-only Model Context Protocol (MCP) server for document search and retrieval.
"""

from mcp_server.server import create_mcp_app, get_mcp_server

__all__ = ["create_mcp_app", "get_mcp_server"]
