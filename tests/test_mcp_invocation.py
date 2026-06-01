from __future__ import annotations

import asyncio

from app.services.mcp.client_manager import McpClientManager
from app.services.mcp.config import McpConfig, McpGlobalConfig, McpServerConfig
from app.services.mcp.exceptions import McpToolError


class FakeConnector:
    def __init__(self, *, delay=0, error=None):
        self.delay = delay
        self.error = error
        self.calls = []
        self.closed = False

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {"content": [{"type": "text", "text": "ok"}]}

    async def disconnect(self):
        self.closed = True


def _server(allowlist=None, timeout_seconds=1):
    return McpServerConfig(
        id="excalidraw",
        label="Excalidraw",
        url="https://mcp.excalidraw.com",
        enabled=True,
        auth_strategy=None,
        auth_secret_env=None,
        timeout_seconds=timeout_seconds,
        tool_allowlist=allowlist or [],
        artifact_capabilities=[],
        health_check_interval_seconds=300,
        tags=[],
    )


class FakeInvokeManager(McpClientManager):
    def __init__(self, config, global_config, connector):
        super().__init__(config=config, global_config=global_config)
        self.connector = connector

    async def get_connector(self, server_id):
        return self.connector


def _manager(connector, *, allowlist=None, timeout_seconds=1, total_budget_seconds=10, enabled=True, invocation_enabled=True):
    config = McpConfig(servers=[_server(allowlist=allowlist, timeout_seconds=timeout_seconds)])
    global_config = McpGlobalConfig(
        enabled=enabled,
        remote_enabled=True,
        tool_invocation_enabled=invocation_enabled,
        total_budget_seconds=total_budget_seconds,
    )
    return FakeInvokeManager(config, global_config, connector)


def test_invoke_tool_success_returns_wrapped_result():
    connector = FakeConnector()
    manager = _manager(connector, allowlist=["draw"])

    result = asyncio.run(manager.invoke_tool("excalidraw", "draw", {"prompt": "diagram"}, user="alice", session="s1"))

    assert result.status == "success"
    assert result.server_id == "excalidraw"
    assert result.tool_name == "draw"
    assert result.raw_content == {"content": [{"type": "text", "text": "ok"}]}
    assert result.error is None
    assert result.duration_ms >= 0
    assert connector.calls == [("draw", {"prompt": "diagram"})]


def test_invoke_tool_rejects_disallowed_tool_before_remote_call():
    connector = FakeConnector()
    manager = _manager(connector, allowlist=["draw"])

    result = asyncio.run(manager.invoke_tool("excalidraw", "erase", {}))

    assert result.status == "allowlist_rejected"
    assert "not allowed" in result.error
    assert connector.calls == []


def test_invoke_tool_returns_distinct_timeout_status():
    connector = FakeConnector(delay=0.05)
    manager = _manager(connector, allowlist=["draw"], timeout_seconds=0.01)

    result = asyncio.run(manager.invoke_tool("excalidraw", "draw", {}))

    assert result.status == "timeout"
    assert "timed out" in result.error.lower()


def test_invoke_tool_wraps_tool_errors():
    connector = FakeConnector(error=McpToolError("remote bad"))
    manager = _manager(connector, allowlist=["draw"])

    result = asyncio.run(manager.invoke_tool("excalidraw", "draw", {}))

    assert result.status == "tool_error"
    assert "remote bad" in result.error


def test_invoke_tool_enforces_total_budget_per_turn():
    connector = FakeConnector()
    manager = _manager(connector, allowlist=["draw"], total_budget_seconds=0.001)

    first = asyncio.run(manager.invoke_tool("excalidraw", "draw", {}, turn_id="turn-1"))
    second = asyncio.run(manager.invoke_tool("excalidraw", "draw", {}, turn_id="turn-1"))

    assert first.status == "success"
    assert second.status == "budget_exceeded"


def test_disabled_mcp_returns_disabled_status():
    connector = FakeConnector()
    manager = _manager(connector, enabled=False)

    result = asyncio.run(manager.invoke_tool("excalidraw", "draw", {}))

    assert result.status == "disabled"
    assert connector.calls == []
