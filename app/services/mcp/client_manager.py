"""MCP client manager and safe tool invocation."""
from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any, Callable

from app.services.mcp.artifact import McpToolResult
from app.services.mcp.config import McpConfig, McpGlobalConfig, load_mcp_config
from app.services.mcp.connector import McpRemoteConnector
from app.services.mcp.exceptions import McpConnectionError, McpTimeoutError, McpToolError

logger = logging.getLogger(__name__)


class McpClientManager:
    """Lazy connection pool for configured MCP remote servers."""

    def __init__(
        self,
        *,
        config: McpConfig | None = None,
        global_config: McpGlobalConfig | None = None,
        connector_factory: Callable[..., McpRemoteConnector] | None = None,
    ):
        self.config = config or load_mcp_config(global_config=global_config)
        self.global_config = global_config or self.config.global_config
        self._connector_factory = connector_factory or McpRemoteConnector
        self._connectors: dict[str, McpRemoteConnector] = {}
        self._turn_budget_used_seconds: dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: Any) -> "McpClientManager":
        global_config = McpGlobalConfig.from_settings(settings)
        config = load_mcp_config(
            config_path=getattr(settings, "mcp_config_file", None) or None,
            global_config=global_config,
        )
        return cls(config=config, global_config=global_config)

    @property
    def enabled(self) -> bool:
        return bool(self.global_config.enabled and self.global_config.remote_enabled)

    def get_server_config(self, server_id: str):
        return self.config.get_server(server_id)

    async def get_connector(self, server_id: str) -> McpRemoteConnector:
        if not self.enabled:
            raise McpConnectionError("MCP is disabled")
        server = self.config.get_server(server_id)
        if server is None:
            raise McpConnectionError(f"Unknown or disabled MCP server: {server_id}")
        connector = self._connectors.get(server_id)
        if connector is None:
            connector = self._connector_factory(
                server,
                max_retries=self.global_config.max_retries,
            )
            self._connectors[server_id] = connector
        return connector

    async def shutdown_all(self):
        connectors = list(self._connectors.values())
        self._connectors.clear()
        for connector in connectors:
            try:
                await connector.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("MCP connector shutdown failed server_id=%s", connector.server_config.id)

    def is_tool_allowed(self, server_id: str, tool_name: str) -> bool:
        server = self.config.get_server(server_id)
        return bool(server and server.is_tool_allowed(tool_name))

    def _budget_remaining(self, turn_id: str | None) -> float:
        if not turn_id:
            return self.global_config.total_budget_seconds
        used = self._turn_budget_used_seconds.get(turn_id, 0.0)
        return self.global_config.total_budget_seconds - used

    def _record_budget(self, turn_id: str | None, elapsed_seconds: float):
        if not turn_id:
            return
        # Keep a 1ms accounting quantum so sub-millisecond calls still consume budget.
        elapsed = max(float(elapsed_seconds), 0.001)
        self._turn_budget_used_seconds[turn_id] = self._turn_budget_used_seconds.get(turn_id, 0.0) + elapsed

    def _result(
        self,
        *,
        server_id: str,
        tool_name: str,
        status: str,
        started_at: float,
        raw_content: Any = None,
        error: str | None = None,
    ) -> McpToolResult:
        return McpToolResult(
            server_id=server_id,
            tool_name=tool_name,
            status=status,
            duration_ms=max(0, round((perf_counter() - started_at) * 1000)),
            raw_content=raw_content,
            error=error,
        )

    async def invoke_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None,
        *,
        user: str | None = None,
        session: str | None = None,
        turn_id: str | None = None,
    ) -> McpToolResult:
        started = perf_counter()
        audit_base = {
            "server_id": server_id,
            "tool_name": tool_name,
            "user": user or "",
            "session": session or "",
        }
        logger.info(
            "MCP_TOOL_INVOKE server=%s tool=%s user=%s session=%s",
            server_id,
            tool_name,
            user or "",
            session or "",
        )

        if not self.enabled or not self.global_config.tool_invocation_enabled:
            result = self._result(server_id=server_id, tool_name=tool_name, status="disabled", started_at=started, error="MCP is disabled")
            logger.info("MCP_TOOL_RESULT server=%s tool=%s status=%s duration_ms=%s", server_id, tool_name, result.status, result.duration_ms)
            return result

        server = self.config.get_server(server_id)
        if server is None:
            result = self._result(server_id=server_id, tool_name=tool_name, status="connection_error", started_at=started, error="Unknown or disabled MCP server")
            logger.info("MCP_TOOL_RESULT server=%s tool=%s status=%s duration_ms=%s", server_id, tool_name, result.status, result.duration_ms)
            return result

        if not server.is_tool_allowed(tool_name):
            result = self._result(
                server_id=server_id,
                tool_name=tool_name,
                status="allowlist_rejected",
                started_at=started,
                error=f"Tool {tool_name!r} is not allowed for MCP server {server_id!r}",
            )
            logger.warning("MCP_TOOL_RESULT server=%s tool=%s status=%s duration_ms=%s", server_id, tool_name, result.status, result.duration_ms)
            return result

        remaining = self._budget_remaining(turn_id)
        if remaining <= 0:
            result = self._result(server_id=server_id, tool_name=tool_name, status="budget_exceeded", started_at=started, error="Total MCP budget exceeded for this turn")
            logger.warning("MCP_TOOL_RESULT server=%s tool=%s status=%s duration_ms=%s", server_id, tool_name, result.status, result.duration_ms)
            return result

        timeout_seconds = min(float(server.timeout_seconds), remaining)
        try:
            connector = await self.get_connector(server_id)
            raw = await asyncio.wait_for(
                connector.call_tool(tool_name, arguments or {}),
                timeout=timeout_seconds,
            )
            result = self._result(server_id=server_id, tool_name=tool_name, status="success", started_at=started, raw_content=raw)
        except asyncio.TimeoutError:
            result = self._result(server_id=server_id, tool_name=tool_name, status="timeout", started_at=started, error=f"MCP tool timed out after {timeout_seconds:g}s")
        except McpTimeoutError as exc:
            result = self._result(server_id=server_id, tool_name=tool_name, status="timeout", started_at=started, error=str(exc))
        except McpConnectionError as exc:
            result = self._result(server_id=server_id, tool_name=tool_name, status="connection_error", started_at=started, error=str(exc))
        except McpToolError as exc:
            result = self._result(server_id=server_id, tool_name=tool_name, status="tool_error", started_at=started, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            result = self._result(server_id=server_id, tool_name=tool_name, status="error", started_at=started, error=type(exc).__name__)

        self._record_budget(turn_id, perf_counter() - started)
        logger.info(
            "MCP_TOOL_RESULT server=%s tool=%s status=%s duration_ms=%s",
            audit_base["server_id"],
            audit_base["tool_name"],
            result.status,
            result.duration_ms,
        )
        return result
