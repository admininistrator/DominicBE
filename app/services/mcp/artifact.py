"""MCP artifact and tool-result models plus public-output sanitization helpers."""
from __future__ import annotations

import os
import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

DEFAULT_MAX_ARTIFACT_CONTENT_BYTES = 500 * 1024
_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_SCRIPT_TAG_RE = re.compile(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_EVENT_HANDLER_RE = re.compile(r"\s+on[a-zA-Z][\w:-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_JS_URL_RE = re.compile(r"javascript\s*:", re.IGNORECASE)


class Artifact(BaseModel):
    id: str
    type: str
    title: str
    mime_type: str | None = None
    content: str | None = None
    url: str | None = None
    preview_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_server_id: str
    tool_name: str
    safe: bool = False


McpToolStatus = Literal[
    "success",
    "disabled",
    "allowlist_rejected",
    "budget_exceeded",
    "connection_error",
    "timeout",
    "tool_error",
    "error",
]


class McpToolResult(BaseModel):
    server_id: str
    tool_name: str
    status: McpToolStatus
    duration_ms: int
    raw_content: Any = None
    error: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)


def configured_max_artifact_content_bytes(default: int = DEFAULT_MAX_ARTIFACT_CONTENT_BYTES) -> int:
    """Return max inline artifact size, overridable by environment for deployments/tests."""

    value = os.environ.get("MCP_MAX_ARTIFACT_CONTENT_BYTES")
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def is_safe_https_url(value: str | None) -> bool:
    """Validate URLs that are safe to include in public chat artifact responses."""

    if not value or not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped or any(sep in stripped for sep in ("\x00", "\r", "\n")):
        return False
    parsed = urlparse(stripped)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    hostname = (parsed.hostname or "").strip().lower()
    if hostname in _LOCAL_HOSTNAMES or hostname.endswith(".localhost"):
        return False
    return True


def sanitize_svg_content(content: str) -> str:
    """Remove active content from SVG before it is marked safe."""

    sanitized = _SCRIPT_TAG_RE.sub("", content or "")
    sanitized = _EVENT_HANDLER_RE.sub("", sanitized)
    sanitized = _JS_URL_RE.sub("", sanitized)
    return sanitized


def sanitize_artifact(
    artifact: Artifact,
    *,
    max_content_bytes: int | None = None,
) -> Artifact | None:
    """Validate and sanitize an artifact before it can be returned to a client.

    Returns a copy with ``safe=True`` only when URL and inline-content checks pass.
    Unsafe URL artifacts are dropped by returning ``None``. Oversized inline content
    is removed when a safe URL/preview URL remains as a link-only fallback.
    """

    limit = max_content_bytes or configured_max_artifact_content_bytes()
    data = artifact.model_dump()
    metadata = dict(data.get("metadata") or {})

    for field_name in ("url", "preview_url"):
        url = data.get(field_name)
        if url is not None:
            if not is_safe_https_url(url):
                return None
            data[field_name] = url.strip()

    content = data.get("content")
    if content is not None:
        text = str(content)
        if (data.get("mime_type") or "").lower() == "image/svg+xml" or text.lstrip().lower().startswith("<svg"):
            text = sanitize_svg_content(text)
            data["mime_type"] = data.get("mime_type") or "image/svg+xml"
        if len(text.encode("utf-8")) > limit:
            if data.get("url") or data.get("preview_url"):
                data["content"] = None
                metadata["content_dropped_reason"] = "size_limit_exceeded"
                metadata["max_content_bytes"] = limit
            else:
                return None
        else:
            data["content"] = text

    data["metadata"] = metadata
    data["safe"] = True
    return Artifact(**data)
