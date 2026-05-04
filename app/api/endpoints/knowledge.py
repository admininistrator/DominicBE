"""Knowledge base router – upload, list, chunks, jobs, reindex."""
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_admin
from app.api.error_handling import raise_internal_server_error
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import get_logger
from app.core.security import validate_username_policy
from app.crud import crud_knowledge
from app.models.chat_models import User
from app.schemas.knowledge_schemas import (
    AuditLogResponse,
    CostMetricsResponse,
    IngestionResult,
    IngestionJobResponse,
    KnowledgeChunkResponse,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentResponse,
    RetrievalAnalyticsResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    ThreeStorageBackfillRequest,
    ThreeStorageBackfillResponse,
)
from app.services.retrieval_service import search_knowledge
from app.services import vector_store
from app.services.knowledge_service import (
    backfill_documents_storage,
    create_document_record,
    delete_document_storage,
    extract_text_from_file,
    ingest_document,
    ingest_uploaded_file,
    reindex_document,
    run_indexing_pipeline,
)

router = APIRouter()
logger = get_logger(__name__)


def _validate_optional_username_filter(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_username_policy(value, field_name=field_name, min_length=1, max_length=255)


def _audit(db: Session, actor: str, action: str, **kwargs):
    """Best-effort audit log write – never raises."""
    if not settings.audit_log_enabled:
        return
    try:
        crud_knowledge.create_audit_log(db, actor, action, **kwargs)
    except Exception:
        pass


@router.get("/admin/analytics", response_model=RetrievalAnalyticsResponse)
def get_admin_analytics(
    username: str | None = None,
    recent_limit: int = 20,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    try:
        normalized_username = _validate_optional_username_filter(username, field_name="username")
        analytics = crud_knowledge.get_retrieval_analytics(
            db,
            username=normalized_username,
            recent_limit=max(1, min(recent_limit, 100)),
        )
        return RetrievalAnalyticsResponse(**analytics)
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.get_admin_analytics", exc=e)


@router.post("/admin/backfill-three-storage", response_model=ThreeStorageBackfillResponse)
def backfill_three_storage(
    request: ThreeStorageBackfillRequest,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    try:
        summary = backfill_documents_storage(
            db,
            document_ids=request.document_ids,
            owner_username=request.owner_username,
            limit=request.limit,
            write_object_artifacts=request.write_object_artifacts,
            upsert_vectors=request.upsert_vectors,
            write_source_manifest=request.write_source_manifest,
            fail_fast=request.fail_fast,
        )
        _audit(
            db,
            admin_user.username,
            "knowledge.backfill_three_storage",
            resource_type="knowledge",
            detail_json={
                "document_ids": request.document_ids,
                "owner_username": request.owner_username,
                "limit": request.limit,
                "write_object_artifacts": request.write_object_artifacts,
                "upsert_vectors": request.upsert_vectors,
                "write_source_manifest": request.write_source_manifest,
                "fail_fast": request.fail_fast,
                "success_count": summary["success_count"],
                "error_count": summary["error_count"],
                "total_vector_points": summary["total_vector_points"],
            },
        )
        return ThreeStorageBackfillResponse(**summary)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.backfill_three_storage", exc=e)


# ---------------------------------------------------------------------------
# Upload file
# ---------------------------------------------------------------------------

@router.post("/documents/upload", response_model=IngestionResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_index: bool = False,
    session_id: int | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a file (txt, md, pdf, docx) and run ingestion pipeline.

    Pass ``async_index=true`` to return immediately with ``status=pending`` and
    run chunking + embedding in a background task.  Poll ``GET /jobs/{job_id}``
    to track progress.

    Pass ``session_id`` to associate the document with a specific chat session.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    max_bytes = settings.knowledge_max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.knowledge_max_upload_size_mb} MB).",
        )

    try:
        if async_index:
            raw_text = extract_text_from_file(content, file.filename, file.content_type)
            record = create_document_record(
                db=db,
                owner_username=current_user.username,
                title=file.filename,
                raw_text=raw_text,
                source_type="upload",
                source_uri=file.filename,
                mime_type=file.content_type,
                session_id=session_id,
                source_bytes=content,
                source_filename=file.filename,
            )
            background_tasks.add_task(
                run_indexing_pipeline,
                record["document_id"],
                record["job_id"],
                SessionLocal,
            )
            _audit(db, current_user.username, "document.upload",
                   resource_type="document", resource_id=record["document_id"],
                   detail_json={"filename": file.filename, "async": True, "session_id": session_id})
            return IngestionResult(**record)
        else:
            result = ingest_uploaded_file(
                db=db,
                owner_username=current_user.username,
                filename=file.filename,
                content=content,
                mime_type=file.content_type,
                session_id=session_id,
            )
            _audit(db, current_user.username, "document.upload",
                   resource_type="document", resource_id=result["document_id"],
                   detail_json={"filename": file.filename, "async": False, "session_id": session_id})
            return IngestionResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.upload_document", exc=e)


# ---------------------------------------------------------------------------
# Create from raw text
# ---------------------------------------------------------------------------

@router.post("/documents/ingest", response_model=IngestionResult, status_code=status.HTTP_201_CREATED)
def ingest_text(
    request: KnowledgeDocumentCreateRequest,
    background_tasks: BackgroundTasks,
    async_index: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ingest a document from raw text.

    Pass ``async_index=true`` to return immediately and run indexing in the
    background.  Poll ``GET /jobs/{job_id}`` to track progress.

    Pass ``session_id`` in the request body to associate with a chat session.
    """
    if not request.raw_text and request.source_type == "text":
        raise HTTPException(status_code=400, detail="raw_text is required for source_type='text'.")

    try:
        if async_index:
            record = create_document_record(
                db=db,
                owner_username=current_user.username,
                title=request.title,
                raw_text=request.raw_text or "",
                source_type=request.source_type,
                source_uri=request.source_uri,
                mime_type=request.mime_type,
                metadata=request.metadata,
                session_id=request.session_id,
            )
            background_tasks.add_task(
                run_indexing_pipeline,
                record["document_id"],
                record["job_id"],
                SessionLocal,
            )
            return IngestionResult(**record)
        else:
            result = ingest_document(
                db=db,
                owner_username=current_user.username,
                title=request.title,
                raw_text=request.raw_text or "",
                source_type=request.source_type,
                source_uri=request.source_uri,
                mime_type=request.mime_type,
                metadata=request.metadata,
                session_id=request.session_id,
            )
            _audit(db, current_user.username, "document.ingest",
                   resource_type="document", resource_id=result["document_id"],
                   detail_json={"title": request.title, "async": False, "session_id": request.session_id})
            return IngestionResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.ingest_text", exc=e)


# ---------------------------------------------------------------------------
# Search indexed chunks
# ---------------------------------------------------------------------------

@router.post("/search", response_model=KnowledgeSearchResponse)
def search_documents(
    request: KnowledgeSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = search_knowledge(
            db=db,
            owner_username=current_user.username,
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
        return KnowledgeSearchResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.search_documents", exc=e)


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=list[KnowledgeDocumentResponse])
def list_documents(
    skip: int = 0,
    limit: int = 50,
    session_id: int | None = None,
    session_filter: str = "all",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List knowledge documents.

    ``session_filter`` values:
    - ``all`` (default): return all documents
    - ``session``: return only documents for the given ``session_id``
    - ``global``: return only documents with no session (global/legacy)
    """
    docs = crud_knowledge.list_documents(
        db, current_user.username, skip, limit,
        session_id=session_id,
        session_id_filter=session_filter,
    )
    return docs


# ---------------------------------------------------------------------------
# Get document detail
# ---------------------------------------------------------------------------

@router.get("/documents/{doc_id}", response_model=KnowledgeDocumentResponse)
def get_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = crud_knowledge.get_document(db, doc_id)
    if not doc or doc.owner_username != current_user.username:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


# ---------------------------------------------------------------------------
# List chunks for a document
# ---------------------------------------------------------------------------

@router.get("/documents/{doc_id}/chunks", response_model=list[KnowledgeChunkResponse])
def list_chunks(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = crud_knowledge.get_document(db, doc_id)
    if not doc or doc.owner_username != current_user.username:
        raise HTTPException(status_code=404, detail="Document not found.")
    return crud_knowledge.get_chunks_by_document(db, doc_id)


@router.get("/documents/{doc_id}/jobs", response_model=list[IngestionJobResponse])
def list_document_jobs(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = crud_knowledge.get_document(db, doc_id)
    if not doc or doc.owner_username != current_user.username:
        raise HTTPException(status_code=404, detail="Document not found.")
    return crud_knowledge.list_ingestion_jobs(db, doc_id)


# ---------------------------------------------------------------------------
# Reindex a document
# ---------------------------------------------------------------------------

@router.post("/documents/{doc_id}/reindex", response_model=IngestionResult)
def reindex(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = crud_knowledge.get_document(db, doc_id)
    if not doc or doc.owner_username != current_user.username:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        result = reindex_document(db, doc_id)
        _audit(db, current_user.username, "document.reindex",
               resource_type="document", resource_id=doc_id)
        return IngestionResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.reindex", exc=e)


# ---------------------------------------------------------------------------
# Delete a document (cascades chunks + jobs)
# ---------------------------------------------------------------------------

@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = crud_knowledge.get_document(db, doc_id)
    if not doc or doc.owner_username != current_user.username:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        cleanup = delete_document_storage(
            db,
            doc_id,
            delete_object_artifacts=False,
            delete_vectors=True,
            hard_delete=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.delete_document", exc=e)
    _audit(db, current_user.username, "document.delete",
           resource_type="document", resource_id=doc_id,
           detail_json={
               "title": doc.title,
               "soft_delete": True,
               "vector_store": cleanup["vector_store"],
               "object_storage": cleanup["object_storage"],
           })


# ---------------------------------------------------------------------------
# Ingestion job status
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = crud_knowledge.get_ingestion_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    # Verify ownership via document
    doc = crud_knowledge.get_document(db, job.document_id)
    if not doc or doc.owner_username != current_user.username:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


# ---------------------------------------------------------------------------
# Phase 6: Admin cost / usage dashboard
# ---------------------------------------------------------------------------

@router.get("/admin/cost", response_model=CostMetricsResponse)
def get_cost_metrics(
    username: str | None = None,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: aggregated token usage + retrieval latency cost metrics."""
    try:
        normalized_username = _validate_optional_username_filter(username, field_name="username")
        data = crud_knowledge.get_cost_metrics(db, username=normalized_username)
        return CostMetricsResponse(**data)
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.get_cost_metrics", exc=e)


# ---------------------------------------------------------------------------
# Phase 6: Audit log endpoint
# ---------------------------------------------------------------------------

@router.get("/admin/audit-logs", response_model=list[AuditLogResponse])
def get_audit_logs(
    actor_username: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    skip: int = 0,
    limit: int = 100,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: query audit trail."""
    try:
        normalized_actor_username = _validate_optional_username_filter(
            actor_username,
            field_name="actor_username",
        )
        logs = crud_knowledge.list_audit_logs(
            db,
            actor_username=normalized_actor_username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            skip=max(0, skip),
            limit=max(1, min(limit, 500)),
        )
        return logs
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.get_audit_logs", exc=e)


# ---------------------------------------------------------------------------
# Phase 6: Admin hard-delete (permanent)
# ---------------------------------------------------------------------------

@router.delete("/admin/documents/{doc_id}/hard-delete", status_code=status.HTTP_204_NO_CONTENT)
def hard_delete_document(
    doc_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: permanently delete a document and all related records."""
    try:
        cleanup = delete_document_storage(
            db,
            doc_id,
            delete_object_artifacts=True,
            delete_vectors=True,
            hard_delete=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="knowledge.hard_delete_document", exc=e)
    _audit(db, _admin.username, "document.hard_delete",
           resource_type="document", resource_id=doc_id,
           detail_json={
               "title": cleanup["title"],
               "owner": cleanup["owner_username"],
               "object_storage": cleanup["object_storage"],
               "object_storage_deleted_keys": cleanup["object_storage_deleted_keys"],
               "vector_store": cleanup["vector_store"],
           })


