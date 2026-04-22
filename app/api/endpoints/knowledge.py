"""Knowledge base router – upload, list, chunks, jobs, reindex."""
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_admin
from app.core.config import settings
from app.core.database import SessionLocal
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
)
from app.services.retrieval_service import search_knowledge
from app.services.knowledge_service import (
    create_document_record,
    extract_text_from_file,
    ingest_document,
    ingest_uploaded_file,
    reindex_document,
    run_indexing_pipeline,
)

router = APIRouter()


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
        analytics = crud_knowledge.get_retrieval_analytics(
            db,
            username=(username or None),
            recent_limit=max(1, min(recent_limit, 100)),
        )
        return RetrievalAnalyticsResponse(**analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Upload file
# ---------------------------------------------------------------------------

@router.post("/documents/upload", response_model=IngestionResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_index: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a file (txt, md, pdf, docx) and run ingestion pipeline.

    Pass ``async_index=true`` to return immediately with ``status=pending`` and
    run chunking + embedding in a background task.  Poll ``GET /jobs/{job_id}``
    to track progress.
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
            )
            background_tasks.add_task(
                run_indexing_pipeline,
                record["document_id"],
                record["job_id"],
                SessionLocal,
            )
            _audit(db, current_user.username, "document.upload",
                   resource_type="document", resource_id=record["document_id"],
                   detail_json={"filename": file.filename, "async": True})
            return IngestionResult(**record)
        else:
            result = ingest_uploaded_file(
                db=db,
                owner_username=current_user.username,
                filename=file.filename,
                content=content,
                mime_type=file.content_type,
            )
            _audit(db, current_user.username, "document.upload",
                   resource_type="document", resource_id=result["document_id"],
                   detail_json={"filename": file.filename, "async": False})
            return IngestionResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            )
            _audit(db, current_user.username, "document.ingest",
                   resource_type="document", resource_id=result["document_id"],
                   detail_json={"title": request.title, "async": False})
            return IngestionResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=list[KnowledgeDocumentResponse])
def list_documents(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    docs = crud_knowledge.list_documents(db, current_user.username, skip, limit)
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
        raise HTTPException(status_code=500, detail=str(e))


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
    crud_knowledge.delete_document(db, doc_id)
    _audit(db, current_user.username, "document.delete",
           resource_type="document", resource_id=doc_id,
           detail_json={"title": doc.title, "soft_delete": True})


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
        data = crud_knowledge.get_cost_metrics(db, username=username or None)
        return CostMetricsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        logs = crud_knowledge.list_audit_logs(
            db,
            actor_username=actor_username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            skip=max(0, skip),
            limit=max(1, min(limit, 500)),
        )
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    from app.models.knowledge_models import KnowledgeDocument as KDModel
    doc_raw = db.query(KDModel).filter(KDModel.id == doc_id).first()
    if not doc_raw:
        raise HTTPException(status_code=404, detail="Document not found.")
    title = doc_raw.title
    owner = doc_raw.owner_username
    crud_knowledge.hard_delete_document(db, doc_id)
    _audit(db, _admin.username, "document.hard_delete",
           resource_type="document", resource_id=doc_id,
           detail_json={"title": title, "owner": owner})


