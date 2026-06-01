"""Adapter registry for MCP result normalization."""
from __future__ import annotations

from typing import Any

from app.services.mcp.adapters.base import BaseResultAdapter, GenericResultAdapter
from app.services.mcp.adapters.excalidraw import ExcalidrawResultAdapter

ADAPTER_REGISTRY: dict[str, type[BaseResultAdapter]] = {
    "excalidraw": ExcalidrawResultAdapter,
}


def get_result_adapter(
    server_id: str,
    *,
    tool_name: str = "",
    max_content_bytes: int | None = None,
) -> BaseResultAdapter:
    adapter_cls = ADAPTER_REGISTRY.get(server_id)
    if adapter_cls is None:
        return GenericResultAdapter(server_id=server_id, tool_name=tool_name, max_content_bytes=max_content_bytes)
    return adapter_cls(tool_name=tool_name, max_content_bytes=max_content_bytes)


def normalize_tool_result(
    server_id: str,
    tool_name: str,
    raw_result: Any,
    *,
    max_content_bytes: int | None = None,
):
    return get_result_adapter(
        server_id,
        tool_name=tool_name,
        max_content_bytes=max_content_bytes,
    ).normalize(raw_result)


__all__ = [
    "ADAPTER_REGISTRY",
    "BaseResultAdapter",
    "ExcalidrawResultAdapter",
    "GenericResultAdapter",
    "get_result_adapter",
    "normalize_tool_result",
]
