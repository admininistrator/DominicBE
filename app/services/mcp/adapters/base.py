"""Base MCP result adapters for normalized public artifacts."""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

from app.services.mcp.artifact import Artifact, sanitize_artifact


def artifact_id(prefix: str = "art") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class BaseResultAdapter(ABC):
    """Interface for converting raw MCP tool results to normalized artifacts."""

    server_id = "generic"

    def __init__(self, *, tool_name: str = "", max_content_bytes: int | None = None):
        self.tool_name = tool_name or "unknown"
        self.max_content_bytes = max_content_bytes

    @abstractmethod
    def normalize(self, raw_result: Any) -> list[Artifact]:
        """Normalize one MCP raw result into zero or more sanitized artifacts."""

    def _sanitize(self, artifact: Artifact) -> Artifact | None:
        return sanitize_artifact(artifact, max_content_bytes=self.max_content_bytes)


class GenericResultAdapter(BaseResultAdapter):
    """Safe fallback adapter for MCP servers without a custom result adapter."""

    server_id = "generic"

    def __init__(self, *, server_id: str = "generic", tool_name: str = "", max_content_bytes: int | None = None):
        super().__init__(tool_name=tool_name, max_content_bytes=max_content_bytes)
        self.server_id = server_id or "generic"

    def normalize(self, raw_result: Any) -> list[Artifact]:
        try:
            if isinstance(raw_result, str):
                content = raw_result
            else:
                content = json.dumps(raw_result, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001 - malformed tool outputs should not escape adapter boundary
            return []

        artifact = Artifact(
            id=artifact_id(),
            type="generic_tool_result",
            title=f"{self.server_id} tool result",
            mime_type="application/json" if not isinstance(raw_result, str) else "text/plain",
            content=content,
            metadata={},
            tool_server_id=self.server_id,
            tool_name=self.tool_name,
        )
        sanitized = self._sanitize(artifact)
        return [sanitized] if sanitized is not None else []
