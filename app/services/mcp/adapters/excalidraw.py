"""Excalidraw MCP result adapter."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from app.services.mcp.adapters.base import BaseResultAdapter, artifact_id
from app.services.mcp.artifact import Artifact, is_safe_https_url

_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+", re.IGNORECASE)
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")


class ExcalidrawResultAdapter(BaseResultAdapter):
    """Normalize Excalidraw MCP outputs into sanitized Artifact objects."""

    server_id = "excalidraw"

    def __init__(self, *, tool_name: str = "", max_content_bytes: int | None = None):
        super().__init__(tool_name=tool_name or "excalidraw-tool", max_content_bytes=max_content_bytes)

    def normalize(self, raw_result: Any) -> list[Artifact]:
        artifacts: list[Artifact] = []
        for item in self._iter_items(raw_result):
            artifact = self._artifact_from_item(item)
            if artifact is None:
                continue
            sanitized = self._sanitize(artifact)
            if sanitized is not None:
                artifacts.append(sanitized)
        return artifacts

    def _iter_items(self, raw_result: Any) -> Iterable[Any]:
        if raw_result is None:
            return []
        dumped = self._model_dump(raw_result)
        if dumped is not None:
            return self._iter_items(dumped)
        if hasattr(raw_result, "raw_content"):
            return self._iter_items(getattr(raw_result, "raw_content"))
        if hasattr(raw_result, "content") and not isinstance(raw_result, dict):
            return self._iter_items(getattr(raw_result, "content"))
        if isinstance(raw_result, dict):
            content = raw_result.get("content")
            if isinstance(content, list):
                return content
            if content is not None:
                return [content]
            # Treat dicts that already look like one content part as one item.
            if any(key in raw_result for key in ("text", "json", "url", "uri", "resource")):
                return [raw_result]
            return []
        if isinstance(raw_result, list):
            return raw_result
        if isinstance(raw_result, str):
            return [raw_result]
        return []

    def _artifact_from_item(self, item: Any) -> Artifact | None:
        if item is None:
            return None
        if isinstance(item, str):
            return self._artifact_from_text(item, mime_type=None, url=None)
        if not isinstance(item, dict):
            item = self._object_to_item_dict(item)
        if not isinstance(item, dict):
            return None

        resource = item.get("resource")
        if resource is not None and not isinstance(resource, dict):
            resource = self._object_to_item_dict(resource)
        if isinstance(resource, dict):
            merged = dict(resource)
            for key in ("type", "mimeType", "mime_type"):
                if key in item and key not in merged:
                    merged[key] = item[key]
            return self._artifact_from_item(merged)

        mime_type = item.get("mimeType") or item.get("mime_type") or item.get("mime")
        part_type = str(item.get("type") or "").lower()
        url = item.get("url") or item.get("uri") or item.get("href")
        preview_url = item.get("preview_url") or item.get("previewUrl")
        title = item.get("title") or "Excalidraw artifact"

        if part_type == "image" or (isinstance(mime_type, str) and mime_type.lower().startswith("image/")):
            content = item.get("text") or item.get("data") or item.get("content")
            return self._build_artifact(
                artifact_type="image",
                title=str(title if title != "Excalidraw artifact" else "Excalidraw image"),
                mime_type=mime_type or "image/png",
                content=str(content) if content is not None else None,
                url=str(url).strip() if url else None,
                preview_url=str(preview_url).strip() if preview_url else None,
                metadata={"source_type": part_type or "image"},
            )

        if "json" in item:
            try:
                json_content = json.dumps(item["json"], ensure_ascii=False)
            except TypeError:
                return None
            return self._build_artifact(
                artifact_type="excalidraw",
                title=str(title),
                mime_type="application/json",
                content=json_content,
                url=str(url).strip() if url else None,
                preview_url=str(preview_url).strip() if preview_url else None,
                metadata={"source_type": part_type or "json"},
            )

        text = item.get("text") or item.get("content")
        if text is not None:
            return self._artifact_from_text(str(text), mime_type=mime_type, url=str(url).strip() if url else None)

        if url:
            return self._artifact_from_url(str(url), mime_type=mime_type)
        return None

    def _artifact_from_text(self, text: str, *, mime_type: str | None, url: str | None) -> Artifact | None:
        stripped = (text or "").strip()
        if not stripped:
            return None
        normalized_mime = (mime_type or "").lower()
        if normalized_mime == "image/svg+xml" or stripped.lower().startswith("<svg"):
            return self._build_artifact(
                artifact_type="image",
                title="Excalidraw SVG export",
                mime_type="image/svg+xml",
                content=stripped,
                url=url,
                metadata={"source_type": "svg"},
            )

        if is_safe_https_url(stripped):
            return self._artifact_from_url(stripped, mime_type=mime_type)

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return self._build_artifact(
                artifact_type="excalidraw",
                title="Excalidraw scene",
                mime_type="application/json",
                content=json.dumps(parsed, ensure_ascii=False),
                url=url,
                metadata={"source_type": "json_text"},
            )

        # Explicitly reject unsafe URL-like/local-path payloads instead of turning
        # them into public text artifacts.
        if ":" in stripped.split()[0] or stripped.startswith(("/", "\\")):
            return None

        found_url = _URL_RE.search(stripped)
        if found_url:
            return self._artifact_from_url(found_url.group(0), mime_type=mime_type)

        return None

    def _artifact_from_url(self, url: str, *, mime_type: str | None) -> Artifact | None:
        artifact_type = "excalidraw"
        title = "Excalidraw link"
        lower_url = url.lower()
        normalized_mime = (mime_type or "").lower()
        if normalized_mime.startswith("image/") or lower_url.split("?", 1)[0].endswith(_IMAGE_EXTENSIONS):
            artifact_type = "image"
            title = "Excalidraw image"
        return self._build_artifact(
            artifact_type=artifact_type,
            title=title,
            mime_type=mime_type,
            url=url.strip(),
            metadata={"source_type": "link"},
        )

    def _model_dump(self, value: Any) -> dict[str, Any] | None:
        model_dump = getattr(value, "model_dump", None)
        if not callable(model_dump):
            return None
        try:
            dumped = model_dump()
        except Exception:  # noqa: BLE001 - SDK objects should not escape adapter boundary
            return None
        return dumped if isinstance(dumped, dict) else None

    def _object_to_item_dict(self, value: Any) -> dict[str, Any] | None:
        dumped = self._model_dump(value)
        if dumped is not None:
            return dumped

        data: dict[str, Any] = {}
        for attr in (
            "type",
            "text",
            "content",
            "mimeType",
            "mime_type",
            "mime",
            "url",
            "uri",
            "href",
            "resource",
            "data",
            "json",
            "title",
            "preview_url",
            "previewUrl",
        ):
            try:
                attr_value = getattr(value, attr)
            except Exception:  # noqa: BLE001 - tolerate custom SDK property failures
                continue
            if attr_value is not None:
                data[attr] = attr_value
        return data or None

    def _build_artifact(
        self,
        *,
        artifact_type: str,
        title: str,
        mime_type: str | None = None,
        content: str | None = None,
        url: str | None = None,
        preview_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        return Artifact(
            id=artifact_id(),
            type=artifact_type,
            title=title,
            mime_type=mime_type,
            content=content,
            url=url,
            preview_url=preview_url,
            metadata=metadata or {},
            tool_server_id=self.server_id,
            tool_name=self.tool_name,
        )
