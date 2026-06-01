from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.mcp.adapters import normalize_tool_result
from app.services.mcp.client_manager import McpClientManager
from app.services.mcp.config import McpConfig, McpGlobalConfig, McpServerConfig, load_mcp_config
from app.services.mcp.exceptions import McpConnectionError
from app.services.mcp.tool_registry import McpToolRegistry


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class FakeConnector:
    def __init__(self, *, tools=None, raw=None, error=None):
        self.tools = tools or []
        self.raw = raw or {"content": [{"type": "text", "text": "https://excalidraw.com/#json=mock"}]}
        self.error = error
        self.list_calls = 0
        self.call_args = []

    async def list_tools(self):
        self.list_calls += 1
        if self.error:
            raise self.error
        return self.tools

    async def call_tool(self, name, arguments):
        self.call_args.append((name, arguments))
        if self.error:
            raise self.error
        return self.raw


class FakeManager(McpClientManager):
    def __init__(self, config, connector, *, enabled=True):
        super().__init__(
            config=config,
            global_config=McpGlobalConfig(enabled=enabled, remote_enabled=True, tool_invocation_enabled=True),
        )
        self.connector = connector

    async def get_connector(self, server_id):
        return self.connector


def _excalidraw_server(**overrides):
    data = {
        "id": "excalidraw",
        "label": "Excalidraw Whiteboard",
        "url": "https://mcp.excalidraw.com",
        "enabled": True,
        "auth_strategy": None,
        "auth_secret_env": None,
        "timeout_seconds": 30,
        "tool_allowlist": [],
        "artifact_capabilities": ["excalidraw_json", "link", "svg", "png_url"],
        "health_check_interval_seconds": 300,
        "tags": ["drawing", "diagramming"],
    }
    data.update(overrides)
    return McpServerConfig(**data)


def test_runtime_registry_contains_enabled_excalidraw_server():
    config = load_mcp_config(config_path=BACKEND_ROOT / "config" / "mcp_servers.json", env={})

    server = config.get_server("excalidraw")
    assert server is not None
    assert server.url == "https://mcp.excalidraw.com"
    assert server.enabled is True
    assert server.artifact_capabilities == ["excalidraw_json", "link", "svg", "png_url"]


def test_mock_excalidraw_tool_list_is_available_without_network():
    connector = FakeConnector(
        tools=[
            {
                "name": "create-excalidraw",
                "description": "Create an Excalidraw scene",
                "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
            },
            {
                "name": "export-image",
                "description": "Export an Excalidraw scene",
                "inputSchema": {"type": "object"},
            },
        ]
    )
    manager = FakeManager(McpConfig(servers=[_excalidraw_server()]), connector)
    registry = McpToolRegistry(manager, cache_ttl_seconds=300)

    tools = asyncio.run(registry.get_tools("excalidraw"))

    assert [tool["name"] for tool in tools] == ["create-excalidraw", "export-image"]
    assert all(tool["server_id"] == "excalidraw" for tool in tools)
    assert connector.list_calls == 1


def test_mock_successful_excalidraw_invocation_normalizes_artifacts_without_network():
    connector = FakeConnector(raw={"content": [{"type": "text", "text": "https://excalidraw.com/#json=mock"}]})
    manager = FakeManager(McpConfig(servers=[_excalidraw_server()]), connector)

    result = asyncio.run(manager.invoke_tool("excalidraw", "create-excalidraw", {"prompt": "draw a flow"}))
    artifacts = normalize_tool_result("excalidraw", "create-excalidraw", result.raw_content)

    assert result.status == "success"
    assert len(artifacts) == 1
    assert artifacts[0].type == "excalidraw"
    assert artifacts[0].url == "https://excalidraw.com/#json=mock"
    assert artifacts[0].safe is True


def test_mock_server_unavailable_returns_graceful_connection_error():
    connector = FakeConnector(error=McpConnectionError("server unavailable"))
    manager = FakeManager(McpConfig(servers=[_excalidraw_server()]), connector)

    result = asyncio.run(manager.invoke_tool("excalidraw", "create-excalidraw", {"prompt": "draw"}))

    assert result.status == "connection_error"
    assert "server unavailable" in result.error


def test_mock_excalidraw_timeout_returns_graceful_timeout_error():
    connector = FakeConnector(error=asyncio.TimeoutError())
    manager = FakeManager(McpConfig(servers=[_excalidraw_server()]), connector)

    result = asyncio.run(manager.invoke_tool("excalidraw", "create-excalidraw", {"prompt": "draw"}))

    assert result.status == "timeout"
    assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


def test_mock_malformed_response_normalizes_to_empty_safe_artifacts():
    artifacts = normalize_tool_result("excalidraw", "create-excalidraw", {"unexpected": object()})

    assert artifacts == []
