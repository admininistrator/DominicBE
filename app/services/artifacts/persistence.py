"""Persistence service for chat artifacts.

Phase 2 writes first-class rows in the ``artifacts`` table while preserving the
Phase 1 message-payload JSON contract for backward-compatible history loading.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.chat_models import ChatArtifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    session_id: int
    message_id: int
    kind: str
    title: str
    content_json: str | None = None
    elements_json: str | None = None
    metadata: dict[str, Any] | None = None


class ArtifactPersistenceService(Protocol):
    def save_many(self, db: Session, records: list[ArtifactRecord]) -> list[ArtifactRecord]:
        """Persist artifact records and return the records that were accepted."""

    def list_by_message_ids(self, db: Session, message_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        """Load public artifact responses grouped by message id."""


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return None


def _json_loads_mapping(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_loads_list(value: str | None) -> list[Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, list) else None


def artifact_response_to_record(
    artifact: dict[str, Any],
    *,
    session_id: int,
    message_id: int,
) -> ArtifactRecord | None:
    """Convert a public artifact response dict into a first-class DB record."""

    if not isinstance(artifact, dict):
        return None
    artifact_id = str(artifact.get("id") or "").strip()
    kind = str(artifact.get("kind") or artifact.get("type") or "generic").strip() or "generic"
    title = str(artifact.get("title") or kind).strip() or kind
    if not artifact_id:
        return None

    content = artifact.get("content")
    content_json = str(content) if isinstance(content, str) and content.strip() else None
    elements_json = None
    scene = _json_loads_mapping(content_json)
    if scene and isinstance(scene.get("elements"), list):
        elements_json = _json_dumps(scene["elements"])

    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    metadata = {
        **metadata,
        "type": artifact.get("type"),
        "url": artifact.get("url"),
        "preview_url": artifact.get("preview_url"),
    }

    return ArtifactRecord(
        id=artifact_id,
        session_id=int(session_id),
        message_id=int(message_id),
        kind=kind,
        title=title,
        content_json=content_json,
        elements_json=elements_json,
        metadata=metadata,
    )


def records_from_artifact_responses(
    artifacts: list[dict[str, Any]] | None,
    *,
    session_id: int,
    message_id: int,
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for artifact in artifacts or []:
        record = artifact_response_to_record(artifact, session_id=session_id, message_id=message_id)
        if record is not None:
            records.append(record)
    return records


class SqlArtifactPersistenceService:
    """Best-effort first-class artifact persistence backed by SQLAlchemy."""

    def save_many(self, db: Session, records: list[ArtifactRecord]) -> list[ArtifactRecord]:
        if not records:
            return []
        accepted: list[ArtifactRecord] = []
        try:
            for record in records:
                row = db.get(ChatArtifact, record.id)
                if row is None:
                    row = ChatArtifact(id=record.id)
                    db.add(row)
                row.session_id = record.session_id
                row.message_id = record.message_id
                row.kind = record.kind
                row.title = record.title
                row.content_json = record.content_json
                row.elements_json = record.elements_json
                row.artifact_metadata_json = _json_dumps(record.metadata or {})
                accepted.append(record)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.warning("Failed to persist chat artifacts in first-class table", exc_info=True)
            return []
        return accepted

    def list_by_message_ids(self, db: Session, message_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        normalized_ids = [int(message_id) for message_id in message_ids if message_id is not None]
        if not normalized_ids:
            return {}
        try:
            rows = (
                db.query(ChatArtifact)
                .filter(ChatArtifact.message_id.in_(normalized_ids))
                .order_by(ChatArtifact.created_at.asc(), ChatArtifact.id.asc())
                .all()
            )
        except SQLAlchemyError:
            logger.debug("Failed to load first-class chat artifacts; using message payload fallback", exc_info=True)
            return {}

        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            metadata = _json_loads_mapping(row.artifact_metadata_json) or {}
            artifact_type = str(metadata.pop("type", None) or row.kind or "generic")
            item: dict[str, Any] = {
                "id": row.id,
                "type": artifact_type,
                "title": row.title,
                "metadata": metadata,
            }
            if row.content_json is not None:
                item["content"] = row.content_json
            url = metadata.pop("url", None)
            preview_url = metadata.pop("preview_url", None)
            if url is not None:
                item["url"] = url
            if preview_url is not None:
                item["preview_url"] = preview_url
            # If only elements_json is available, synthesize a minimal scene for
            # the existing native Excalidraw renderer.
            if "content" not in item and row.kind == "excalidraw":
                elements = _json_loads_list(row.elements_json) or []
                item["content"] = _json_dumps({
                    "type": "excalidraw",
                    "version": 2,
                    "source": "DominicChatbot",
                    "elements": elements,
                    "appState": {"viewBackgroundColor": "#ffffff"},
                    "files": {},
                })
            grouped.setdefault(int(row.message_id), []).append(item)
        return grouped


_default_service = SqlArtifactPersistenceService()


def persist_artifact_responses(
    db: Session,
    *,
    session_id: int,
    message_id: int,
    artifacts: list[dict[str, Any]] | None,
    service: ArtifactPersistenceService | None = None,
) -> list[ArtifactRecord]:
    records = records_from_artifact_responses(artifacts, session_id=session_id, message_id=message_id)
    return (service or _default_service).save_many(db, records)


def list_artifacts_by_message_ids(
    db: Session,
    message_ids: list[int],
    *,
    service: ArtifactPersistenceService | None = None,
) -> dict[int, list[dict[str, Any]]]:
    return (service or _default_service).list_by_message_ids(db, message_ids)
