"""MCP tool discovery registry with allowlist filtering and cache."""
from __future__ import annotations

from time import monotonic
from typing import Any

from app.services.mcp.client_manager import McpClientManager


class McpToolRegistry:
    def __init__(self, client_manager: McpClientManager, *, cache_ttl_seconds: float | None = None):
        self.client_manager = client_manager
        self.cache_ttl_seconds = (
            client_manager.global_config.tool_cache_ttl_seconds
            if cache_ttl_seconds is None
            else float(cache_ttl_seconds)
        )
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def clear_cache(self):
        self._cache.clear()

    def _server_config(self, server_id: str):
        return self.client_manager.config.get_server(server_id)

    def _tool_name(self, tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name") or "")
        return str(getattr(tool, "name", ""))

    def _tool_schema(self, tool: Any) -> dict[str, Any] | None:
        if isinstance(tool, dict):
            return tool.get("inputSchema") or tool.get("input_schema")
        return getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)

    def _tool_to_dict(self, server_id: str, tool: Any) -> dict[str, Any]:
        server = self._server_config(server_id)
        if isinstance(tool, dict):
            data = dict(tool)
        else:
            data = {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", ""),
                "inputSchema": self._tool_schema(tool) or {},
            }
        data["server_id"] = server_id
        if server is not None:
            data["server_label"] = server.label
        return data

    def _filter_allowed(self, server_id: str, tools: list[Any]) -> list[dict[str, Any]]:
        server = self._server_config(server_id)
        if server is None:
            return []
        filtered: list[dict[str, Any]] = []
        for tool in tools:
            name = self._tool_name(tool)
            if name and server.is_tool_allowed(name):
                filtered.append(self._tool_to_dict(server_id, tool))
        return filtered

    async def get_tools(self, server_id: str) -> list[dict[str, Any]]:
        if not self.client_manager.enabled:
            return []
        server = self._server_config(server_id)
        if server is None:
            return []
        now = monotonic()
        cached = self._cache.get(server_id)
        if cached and self.cache_ttl_seconds > 0 and now - cached[0] < self.cache_ttl_seconds:
            return list(cached[1])
        connector = await self.client_manager.get_connector(server_id)
        tools = await connector.list_tools()
        filtered = self._filter_allowed(server_id, list(tools or []))
        self._cache[server_id] = (now, filtered)
        return list(filtered)

    async def get_all_tools(self) -> list[dict[str, Any]]:
        if not self.client_manager.enabled:
            return []
        all_tools: list[dict[str, Any]] = []
        for server in self.client_manager.config.servers:
            all_tools.extend(await self.get_tools(server.id))
        return all_tools

    async def is_tool_allowed(self, server_id: str, tool_name: str) -> bool:
        server = self._server_config(server_id)
        return bool(server and server.is_tool_allowed(tool_name))

    async def get_tool_schema(self, server_id: str, tool_name: str) -> dict[str, Any] | None:
        tools = await self.get_tools(server_id)
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool.get("inputSchema") or tool.get("input_schema") or {}
        return None
