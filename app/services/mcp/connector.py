"""Remote MCP connector abstraction."""
from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any, Awaitable, Callable

from app.services.mcp.config import McpServerConfig
from app.services.mcp.exceptions import McpConnectionError, McpTimeoutError, McpToolError

SessionFactory = Callable[[str], Any]


class _SdkSessionWrapper:
    """Small adapter around the official MCP Python SDK StreamableHTTP transport."""

    def __init__(self, url: str, *, headers: dict[str, str], timeout_seconds: float):
        self.url = url
        self.headers = headers
        self.timeout_seconds = timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def initialize(self):
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency installed
            raise McpConnectionError("mcp Python SDK is not installed") from exc

        self._stack = AsyncExitStack()
        read_stream, write_stream, _get_session_id = await self._stack.enter_async_context(
            streamablehttp_client(self.url, headers=self.headers or None, timeout=self.timeout_seconds)
        )
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()

    async def list_tools(self):
        if self._session is None:
            raise McpConnectionError("MCP SDK session is not initialized")
        result = await self._session.list_tools()
        return getattr(result, "tools", result)

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        if self._session is None:
            raise McpConnectionError("MCP SDK session is not initialized")
        return await self._session.call_tool(name, arguments)

    async def close(self):
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None


def _default_session_factory(url: str, *, headers: dict[str, str], timeout_seconds: float):
    return _SdkSessionWrapper(url, headers=headers, timeout_seconds=timeout_seconds)


async def _maybe_await(value: Any) -> Any:
    if isinstance(value, Awaitable):
        return await value
    return value


class McpRemoteConnector:
    """Connects to one configured MCP remote server."""

    def __init__(
        self,
        server_config: McpServerConfig,
        *,
        max_retries: int = 0,
        session_factory: Callable[..., Any] | None = None,
    ):
        self.server_config = server_config
        self.max_retries = max(0, int(max_retries or 0))
        self._session_factory = session_factory or _default_session_factory
        self._session: Any | None = None
        self._connect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._session is not None

    def _auth_headers(self) -> dict[str, str]:
        strategy = (self.server_config.auth_strategy or "").strip().lower()
        if not strategy:
            return {}
        secret_env = (self.server_config.auth_secret_env or "").strip()
        token = os.environ.get(secret_env, "").strip() if secret_env else ""
        if not token:
            return {}
        if strategy == "bearer":
            return {"Authorization": f"Bearer {token}"}
        if strategy == "api_key":
            return {"X-API-Key": token}
        return {}

    async def connect(self):
        if self._session is not None:
            return self._session
        async with self._connect_lock:
            if self._session is not None:
                return self._session
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    session = self._session_factory(
                        self.server_config.url,
                        headers=self._auth_headers(),
                        timeout_seconds=self.server_config.timeout_seconds,
                    )
                    await asyncio.wait_for(
                        _maybe_await(session.initialize()),
                        timeout=self.server_config.timeout_seconds,
                    )
                    self._session = session
                    return session
                except asyncio.TimeoutError as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        raise McpTimeoutError(
                            f"Timed out connecting to MCP server {self.server_config.id}"
                        ) from exc
                except Exception as exc:  # noqa: BLE001 - normalize all transport failures
                    last_error = exc
                    if attempt >= self.max_retries:
                        raise McpConnectionError(
                            f"Failed to connect to MCP server {self.server_config.id}: {type(exc).__name__}"
                        ) from exc
            raise McpConnectionError(f"Failed to connect to MCP server {self.server_config.id}: {last_error}")

    async def disconnect(self):
        session = self._session
        self._session = None
        if session is not None and hasattr(session, "close"):
            await _maybe_await(session.close())

    async def list_tools(self) -> list[Any]:
        session = await self.connect()
        try:
            return await asyncio.wait_for(
                _maybe_await(session.list_tools()),
                timeout=self.server_config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise McpTimeoutError(f"Timed out listing tools for MCP server {self.server_config.id}") from exc
        except (McpConnectionError, McpTimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise McpToolError(f"Failed to list tools for MCP server {self.server_config.id}: {type(exc).__name__}") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        session = await self.connect()
        try:
            return await asyncio.wait_for(
                _maybe_await(session.call_tool(name, arguments)),
                timeout=self.server_config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise McpTimeoutError(
                f"Timed out calling MCP tool {self.server_config.id}.{name}"
            ) from exc
        except (McpConnectionError, McpTimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise McpToolError(f"Failed to call MCP tool {self.server_config.id}.{name}: {exc}") from exc
