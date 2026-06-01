from __future__ import annotations

import asyncio

import pytest

from app.services.mcp.config import McpServerConfig
from app.services.mcp.connector import McpRemoteConnector
from app.services.mcp.exceptions import McpConnectionError, McpTimeoutError, McpToolError


class FakeSession:
    def __init__(self):
        self.initialized = False
        self.closed = False
        self.headers = None
        self.tools = [
            {"name": "draw", "description": "Draw something", "inputSchema": {"type": "object"}},
        ]
        self.tool_result = {"content": [{"type": "text", "text": "ok"}]}

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        if name == "boom":
            raise RuntimeError("remote failed")
        return self.tool_result

    async def close(self):
        self.closed = True


def _server(**overrides):
    data = {
        "id": "excalidraw",
        "label": "Excalidraw",
        "url": "https://mcp.excalidraw.com",
        "enabled": True,
        "auth_strategy": None,
        "auth_secret_env": None,
        "timeout_seconds": 5,
        "tool_allowlist": [],
        "artifact_capabilities": [],
        "health_check_interval_seconds": 300,
        "tags": [],
    }
    data.update(overrides)
    return McpServerConfig(**data)


def test_connector_connects_lists_tools_and_disconnects():
    session = FakeSession()
    connector = McpRemoteConnector(_server(), session_factory=lambda *_args, **_kwargs: session)

    tools = asyncio.run(connector.list_tools())
    asyncio.run(connector.disconnect())

    assert session.initialized is True
    assert tools == session.tools
    assert session.closed is True


def test_connector_applies_bearer_auth_header(monkeypatch):
    monkeypatch.setenv("MCP_EXCALIDRAW_TOKEN", "secret-token")
    session = FakeSession()
    seen = {}

    def factory(url, *, headers, timeout_seconds):
        seen["url"] = url
        seen["headers"] = headers
        seen["timeout_seconds"] = timeout_seconds
        return session

    connector = McpRemoteConnector(
        _server(auth_strategy="bearer", auth_secret_env="MCP_EXCALIDRAW_TOKEN", timeout_seconds=9),
        session_factory=factory,
    )

    asyncio.run(connector.connect())

    assert seen == {
        "url": "https://mcp.excalidraw.com",
        "headers": {"Authorization": "Bearer secret-token"},
        "timeout_seconds": 9,
    }


def test_connector_retries_connection_failures_then_raises():
    attempts = 0

    def factory(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("connect failed")

    connector = McpRemoteConnector(_server(), max_retries=2, session_factory=factory)

    with pytest.raises(McpConnectionError, match="excalidraw"):
        asyncio.run(connector.connect())

    assert attempts == 3


def test_connector_wraps_timeout_and_tool_errors():
    class TimeoutSession(FakeSession):
        async def list_tools(self):
            await asyncio.sleep(0.05)
            return []

    timeout_connector = McpRemoteConnector(
        _server(timeout_seconds=0.01),
        session_factory=lambda *_args, **_kwargs: TimeoutSession(),
    )

    with pytest.raises(McpTimeoutError):
        asyncio.run(timeout_connector.list_tools())

    error_connector = McpRemoteConnector(
        _server(),
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    )

    with pytest.raises(McpToolError, match="boom"):
        asyncio.run(error_connector.call_tool("boom", {}))
