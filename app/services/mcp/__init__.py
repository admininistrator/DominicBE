"""Public API for Dominic backend MCP foundation."""
from app.services.mcp.adapters import GenericResultAdapter, get_result_adapter, normalize_tool_result
from app.services.mcp.artifact import Artifact, McpToolResult, sanitize_artifact
from app.services.mcp.client_manager import McpClientManager
from app.services.mcp.config import McpConfig, McpGlobalConfig, McpServerConfig, load_mcp_config
from app.services.mcp.connector import McpRemoteConnector
from app.services.mcp.tool_registry import McpToolRegistry

__all__ = [
    "Artifact",
    "GenericResultAdapter",
    "McpClientManager",
    "McpConfig",
    "McpGlobalConfig",
    "McpRemoteConnector",
    "McpServerConfig",
    "McpToolRegistry",
    "McpToolResult",
    "get_result_adapter",
    "load_mcp_config",
    "normalize_tool_result",
    "sanitize_artifact",
]
