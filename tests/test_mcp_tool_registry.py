from __future__ import annotations

import asyncio

from app.services.mcp.client_manager import McpClientManager
from app.services.mcp.config import McpConfig, McpGlobalConfig, McpServerConfig
from app.services.mcp.tool_registry import McpToolRegistry


class FakeConnector:
    def __init__(self, tools):
        self.tools = tools
        self.calls = 0

    async def list_tools(self):
        self.calls += 1
        return self.tools


class FakeManager(McpClientManager):
    def __init__(self, config, connectors):
        super().__init__(config=config, global_config=McpGlobalConfig(enabled=True, remote_enabled=True))
        self.fake_connectors = connectors

    async def get_connector(self, server_id):
        return self.fake_connectors[server_id]


def _server(server_id, allowlist=None):
    return McpServerConfig(
        id=server_id,
        label=server_id.title(),
        url=f"https://{server_id}.example/mcp",
        enabled=True,
        auth_strategy=None,
        auth_secret_env=None,
        timeout_seconds=30,
        tool_allowlist=allowlist or [],
        artifact_capabilities=[],
        health_check_interval_seconds=300,
        tags=[],
    )


def test_tool_registry_aggregates_and_filters_by_allowlist():
    config = McpConfig(servers=[_server("one", allowlist=["draw"]), _server("two")])
    one = FakeConnector([
        {"name": "draw", "inputSchema": {"type": "object"}},
        {"name": "erase", "inputSchema": {"type": "object"}},
    ])
    two = FakeConnector([{"name": "search", "inputSchema": {"type": "object"}}])
    registry = McpToolRegistry(FakeManager(config, {"one": one, "two": two}), cache_ttl_seconds=300)

    tools = asyncio.run(registry.get_all_tools())

    assert {(tool["server_id"], tool["name"]) for tool in tools} == {("one", "draw"), ("two", "search")}
    assert asyncio.run(registry.is_tool_allowed("one", "draw")) is True
    assert asyncio.run(registry.is_tool_allowed("one", "erase")) is False
    assert asyncio.run(registry.is_tool_allowed("two", "anything")) is True


def test_tool_registry_caches_tool_discovery_until_invalidated():
    config = McpConfig(servers=[_server("one")])
    connector = FakeConnector([{"name": "draw", "inputSchema": {"type": "object"}}])
    registry = McpToolRegistry(FakeManager(config, {"one": connector}), cache_ttl_seconds=300)

    first = asyncio.run(registry.get_tools("one"))
    second = asyncio.run(registry.get_tools("one"))
    registry.clear_cache()
    third = asyncio.run(registry.get_tools("one"))

    assert first == second == third
    assert connector.calls == 2


def test_tool_registry_get_tool_schema_returns_input_schema():
    config = McpConfig(servers=[_server("one")])
    connector = FakeConnector([{"name": "draw", "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}}}])
    registry = McpToolRegistry(FakeManager(config, {"one": connector}), cache_ttl_seconds=300)

    schema = asyncio.run(registry.get_tool_schema("one", "draw"))

    assert schema == {"type": "object", "properties": {"prompt": {"type": "string"}}}


def test_tool_registry_disabled_global_config_returns_empty():
    config = McpConfig(servers=[_server("one")])
    manager = McpClientManager(config=config, global_config=McpGlobalConfig(enabled=False, remote_enabled=True))
    registry = McpToolRegistry(manager)

    assert asyncio.run(registry.get_all_tools()) == []
