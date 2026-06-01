"""Admin operations router for the dashboard.

This router intentionally returns sanitized operational metadata only. It does
not expose secrets or raw chat message content.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.config import settings
from app.core.database import check_database_health
from app.core.logging import get_logger
from app.crud import crud_chat, crud_knowledge
from app.models.chat_models import ChatSession, Message, User
from app.models.knowledge_models import IngestionJob, KnowledgeChunk, KnowledgeDocument
from app.services.knowledge_service import (
    backfill_documents_storage,
    delete_document_storage,
    reindex_document,
)
from app.services.object_storage import check_object_storage_health
from app.services.rag_core_client import (
    RagCoreClientError,
    get_rag_core_client,
    is_rag_core_api_mode,
)
from app.services.vector_store import check_vector_store_health


router = APIRouter()
logger = get_logger(__name__)


class ConfirmRequest(BaseModel):
    confirm: str = Field(min_length=1, max_length=64)


class AdminBackfillRequest(BaseModel):
    confirm: str = Field(min_length=1, max_length=64)
    document_ids: list[int] = Field(default_factory=list)
    owner_username: str | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
    write_object_artifacts: bool = True
    upsert_vectors: bool = True
    write_source_manifest: bool = True
    fail_fast: bool = False


def _require_confirm(request: ConfirmRequest | AdminBackfillRequest, expected: str) -> None:
    if request.confirm.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Confirmation must be exactly {expected!r}.",
        )


def _audit(
    db: Session,
    request: Request,
    admin: User,
    action: str,
    *,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    result_code: int = status.HTTP_200_OK,
    detail_json: dict[str, Any] | None = None,
) -> None:
    if not settings.audit_log_enabled:
        return
    try:
        crud_knowledge.create_audit_log(
            db,
            admin.username,
            action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            request_id=getattr(request.state, "request_id", None),
            detail_json=detail_json or {},
            result_code=result_code,
        )
    except Exception:
        logger.warning("admin audit write failed action=%s actor=%s", action, admin.username, exc_info=True)


def _rag_core_status() -> dict[str, Any]:
    mode = settings.rag_core_mode
    payload: dict[str, Any] = {
        "mode": mode,
        "api_key_configured": bool(settings.rag_core_api_key),
        "base_url_configured": bool(settings.rag_core_base_url),
    }
    if not is_rag_core_api_mode():
        payload.update({"ok": True, "status": "library_mode"})
        return payload
    try:
        health = get_rag_core_client().health()
        payload.update(
            {
                "ok": bool(health.get("ok")),
                "status": "reachable",
                "service": health.get("service"),
                "dependencies": health.get("dependencies") or {},
            }
        )
    except RagCoreClientError as exc:
        payload.update(
            {
                "ok": False,
                "status": "unreachable",
                "error_type": exc.error_type,
                "status_code": exc.status_code,
            }
        )
    return payload


def _safe_runtime_settings() -> dict[str, Any]:
    return {
        "app": {
            "name": settings.app_name,
            "environment": settings.environment,
            "debug": settings.debug,
        },
        "api": {
            "cors_origins": settings.cors_origins,
            "cors_allow_origin_regex_configured": bool(settings.cors_allow_origin_regex),
            "rate_limit_enabled": settings.rate_limit_enabled,
            "audit_log_enabled": settings.audit_log_enabled,
        },
        "llm": {
            "default_provider": settings.llm_default_provider,
            "default_model": settings.llm_default_model,
            "provider_catalog_configured": bool(
                settings.llm_provider_catalog_json.strip()
                or settings.llm_provider_catalog_file.strip()
            ),
            "context_window": settings.llm_context_window,
            "max_output_tokens": settings.max_output_tokens,
            "vision_enabled": settings.llm_vision_enabled,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "api_type": settings.embedding_api_type,
            "api_key_configured": bool(settings.embedding_api_key),
        },
        "vector_store": {
            "provider": settings.vector_store_provider,
            "collection": settings.vector_store_collection,
            "url_configured": bool(settings.vector_store_url),
            "api_key_configured": bool(settings.vector_store_api_key),
            "prefer_grpc": settings.vector_store_prefer_grpc,
        },
        "object_storage": {
            "provider": settings.object_storage_provider,
            "bucket": settings.object_storage_bucket,
            "endpoint_configured": bool(settings.object_storage_endpoint),
            "access_key_configured": bool(settings.object_storage_access_key),
            "secret_key_configured": bool(settings.object_storage_secret_key),
            "secure": settings.object_storage_secure,
        },
        "rag_core": {
            "mode": settings.rag_core_mode,
            "base_url_configured": bool(settings.rag_core_base_url),
            "api_key_configured": bool(settings.rag_core_api_key),
            "timeout_seconds": settings.rag_core_timeout_seconds,
        },
    }


def _document_payload(db: Session, doc: KnowledgeDocument) -> dict[str, Any]:
    chunk_count = (
        db.query(func.count(KnowledgeChunk.id))
        .filter(KnowledgeChunk.document_id == doc.id)
        .scalar()
        or 0
    )
    latest_job = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == doc.id)
        .order_by(IngestionJob.updated_at.desc(), IngestionJob.id.desc())
        .first()
    )
    return {
        "id": doc.id,
        "owner_username": doc.owner_username,
        "title": doc.title,
        "source_type": doc.source_type,
        "source_uri": doc.source_uri,
        "mime_type": doc.mime_type,
        "status": doc.status,
        "session_id": doc.session_id,
        "checksum": doc.checksum,
        "chunk_count": int(chunk_count),
        "metadata_json": doc.metadata_json or {},
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "deleted_at": doc.deleted_at,
        "latest_job": (
            {
                "id": latest_job.id,
                "status": latest_job.status,
                "error_message": latest_job.error_message,
                "created_at": latest_job.created_at,
                "updated_at": latest_job.updated_at,
            }
            if latest_job
            else None
        ),
    }


def _session_payload(db: Session, session: ChatSession) -> dict[str, Any]:
    aggregate = (
        db.query(
            func.count(Message.id),
            func.coalesce(func.sum(Message.input_tokens), 0),
            func.coalesce(func.sum(Message.output_tokens), 0),
            func.max(Message.id),
        )
        .filter(Message.session_id == session.id, Message.sender_username == session.username)
        .first()
    )
    message_count = int(aggregate[0] or 0)
    input_tokens = int(aggregate[1] or 0)
    output_tokens = int(aggregate[2] or 0)
    latest_message_id = aggregate[3]
    latest_message = (
        db.query(Message).filter(Message.id == latest_message_id).first()
        if latest_message_id
        else None
    )
    return {
        "id": session.id,
        "username": session.username,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": message_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latest_message": (
            {
                "id": latest_message.id,
                "role": latest_message.role,
                "status": latest_message.status,
                "request_id": latest_message.request_id,
                "input_tokens": latest_message.input_tokens,
                "output_tokens": latest_message.output_tokens,
                "created_at": latest_message.created_at,
                "has_images": bool(latest_message.image_payload_json),
                "has_error": bool(latest_message.error_message),
            }
            if latest_message
            else None
        ),
    }


def _admin_health_payload() -> dict[str, Any]:
    dependencies = {
        "database": check_database_health(),
        "object_storage": check_object_storage_health(),
        "vector_store": check_vector_store_health(),
        "rag_core": _rag_core_status(),
    }
    return {
        "ok": all(bool(check.get("ok")) for check in dependencies.values()),
        "service": settings.app_name,
        "dependencies": dependencies,
    }


@router.get("/health")
def admin_health(_admin: User = Depends(require_admin)) -> dict[str, Any]:
    del _admin
    return _admin_health_payload()


@router.get("/settings")
def admin_settings(_admin: User = Depends(require_admin)) -> dict[str, Any]:
    del _admin
    return _safe_runtime_settings()


@router.get("/sessions")
def admin_sessions(
    username: str | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    q: str | None = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del _admin
    query = db.query(ChatSession)
    if username:
        query = query.filter(ChatSession.username == username.strip())
    if q:
        term = f"%{q.strip()}%"
        query = query.filter(or_(ChatSession.title.ilike(term), ChatSession.username.ilike(term)))
    if status_filter:
        query = query.filter(
            db.query(Message.id)
            .filter(Message.session_id == ChatSession.id, Message.status == status_filter.strip())
            .exists()
        )
    total = query.count()
    rows = (
        query.order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {
        "skip": skip,
        "limit": limit,
        "total": int(total),
        "items": [_session_payload(db, row) for row in rows],
    }


@router.get("/knowledge/documents")
def admin_knowledge_documents(
    owner_username: str | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    q: str | None = None,
    include_deleted: bool = False,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del _admin
    query = db.query(KnowledgeDocument)
    if not include_deleted:
        query = query.filter(KnowledgeDocument.deleted_at.is_(None))
    if owner_username:
        query = query.filter(KnowledgeDocument.owner_username == owner_username.strip())
    if status_filter:
        query = query.filter(KnowledgeDocument.status == status_filter.strip())
    if q:
        term = f"%{q.strip()}%"
        query = query.filter(
            or_(
                KnowledgeDocument.title.ilike(term),
                KnowledgeDocument.owner_username.ilike(term),
                KnowledgeDocument.source_uri.ilike(term),
            )
        )
    total = query.count()
    rows = (
        query.order_by(KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {
        "skip": skip,
        "limit": limit,
        "total": int(total),
        "items": [_document_payload(db, row) for row in rows],
    }


@router.get("/overview")
def admin_overview(
    recent_limit: int = Query(default=10, ge=1, le=50),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    del _admin
    users_total = db.query(func.count(User.id)).scalar() or 0
    admin_users = db.query(func.count(User.id)).filter(User.role == "admin").scalar() or 0
    sessions_total = db.query(func.count(ChatSession.id)).scalar() or 0
    messages_total = db.query(func.count(Message.id)).scalar() or 0
    documents_total = db.query(func.count(KnowledgeDocument.id)).filter(KnowledgeDocument.deleted_at.is_(None)).scalar() or 0
    indexed_documents = (
        db.query(func.count(KnowledgeDocument.id))
        .filter(KnowledgeDocument.deleted_at.is_(None), KnowledgeDocument.status == "indexed")
        .scalar()
        or 0
    )
    failed_jobs = db.query(func.count(IngestionJob.id)).filter(IngestionJob.status == "failed").scalar() or 0
    cost = crud_knowledge.get_cost_metrics(db)
    retrieval = crud_knowledge.get_retrieval_analytics(db, recent_limit=recent_limit)
    audit_logs = crud_knowledge.list_audit_logs(db, skip=0, limit=recent_limit)
    return {
        "health": _admin_health_payload(),
        "counts": {
            "users_total": int(users_total),
            "admin_users": int(admin_users),
            "sessions_total": int(sessions_total),
            "messages_total": int(messages_total),
            "documents_total": int(documents_total),
            "indexed_documents": int(indexed_documents),
            "failed_ingestion_jobs": int(failed_jobs),
        },
        "cost": cost,
        "retrieval": retrieval,
        "recent_audit_logs": [
            {
                "id": row.id,
                "actor_username": row.actor_username,
                "action": row.action,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "request_id": row.request_id,
                "result_code": row.result_code,
                "detail_json": row.detail_json or {},
                "created_at": row.created_at,
            }
            for row in audit_logs
        ],
    }


@router.post("/knowledge/documents/{doc_id}/reindex")
def admin_reindex_document(
    doc_id: int,
    request_body: ConfirmRequest,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _require_confirm(request_body, "REINDEX")
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    result = reindex_document(db, doc_id)
    _audit(
        db,
        request,
        admin,
        "admin.document.reindex",
        resource_type="document",
        resource_id=doc_id,
        detail_json={"owner_username": doc.owner_username, "title": doc.title},
    )
    return result


@router.delete("/knowledge/documents/{doc_id}/hard-delete")
def admin_hard_delete_document(
    doc_id: int,
    request_body: ConfirmRequest,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _require_confirm(request_body, "HARD_DELETE")
    try:
        cleanup = delete_document_storage(
            db,
            doc_id,
            delete_object_artifacts=True,
            delete_vectors=True,
            hard_delete=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit(
        db,
        request,
        admin,
        "admin.document.hard_delete",
        resource_type="document",
        resource_id=doc_id,
        detail_json={
            "title": cleanup.get("title"),
            "owner_username": cleanup.get("owner_username"),
            "object_storage": cleanup.get("object_storage"),
            "vector_store": cleanup.get("vector_store"),
        },
    )
    return {"success": True, "cleanup": cleanup}


@router.delete("/sessions/{session_id}")
def admin_delete_session(
    session_id: int,
    request_body: ConfirmRequest,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _require_confirm(request_body, "DELETE_SESSION")
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    username = session.username
    title = session.title
    deleted = crud_chat.delete_chat_session(db, username, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    _audit(
        db,
        request,
        admin,
        "admin.session.delete",
        resource_type="session",
        resource_id=session_id,
        detail_json={"username": username, "title": title},
    )
    return {"success": True, "session_id": session_id}


@router.post("/knowledge/backfill-three-storage")
def admin_backfill_three_storage(
    request_body: AdminBackfillRequest,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _require_confirm(request_body, "BACKFILL")
    summary = backfill_documents_storage(
        db,
        document_ids=request_body.document_ids,
        owner_username=request_body.owner_username,
        limit=request_body.limit,
        write_object_artifacts=request_body.write_object_artifacts,
        upsert_vectors=request_body.upsert_vectors,
        write_source_manifest=request_body.write_source_manifest,
        fail_fast=request_body.fail_fast,
    )
    _audit(
        db,
        request,
        admin,
        "admin.knowledge.backfill_three_storage",
        resource_type="knowledge",
        detail_json={
            "document_ids": request_body.document_ids,
            "owner_username": request_body.owner_username,
            "limit": request_body.limit,
            "success_count": summary.get("success_count"),
            "error_count": summary.get("error_count"),
            "total_vector_points": summary.get("total_vector_points"),
        },
    )
    return summary
