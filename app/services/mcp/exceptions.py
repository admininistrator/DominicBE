"""Exceptions for the MCP client foundation."""
from __future__ import annotations


class McpError(Exception):
    """Base MCP error."""


class McpConfigError(McpError):
    """Raised when MCP configuration is invalid."""


class McpConnectionError(McpError):
    """Raised when an MCP remote cannot be reached or initialized."""


class McpTimeoutError(McpError):
    """Raised when an MCP operation exceeds its timeout budget."""


class McpToolError(McpError):
    """Raised when an MCP tool call or tool discovery operation fails."""


class McpAllowlistError(McpToolError):
    """Raised when a tool invocation is blocked by the configured allowlist."""
