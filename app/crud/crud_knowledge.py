"""CRUD operations for knowledge base (documents, chunks, jobs)."""
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.knowledge_models import (
    AnswerCitation,
    AuditLog,
    IngestionJob,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalEvent,
)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def create_document(
    db: Session,
    owner_username: str,
    title: str,
    source_type: str = "text",
    source_uri: Optional[str] = None,
    mime_type: Optional[str] = None,
    raw_text: Optional[str] = None,
    checksum: Optional[str] = None,
    metadata_json: Optional[dict] = None,
) -> KnowledgeDocument:
    doc = KnowledgeDocument(
        owner_username=owner_username,
        title=title,
        source_type=source_type,
        source_uri=source_uri,
        mime_type=mime_type,
        raw_text=raw_text,
        checksum=checksum,
        metadata_json=metadata_json,
        status="uploaded",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, doc_id: int) -> Optional[KnowledgeDocument]:
    return (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.id == doc_id, KnowledgeDocument.deleted_at.is_(None))
        .first()
    )


def get_document_by_owner_and_checksum(
    db: Session,
    owner_username: str,
    checksum: str,
) -> Optional[KnowledgeDocument]:
    return (
        db.query(KnowledgeDocument)
        .filter(
            KnowledgeDocument.owner_username == owner_username,
            KnowledgeDocument.checksum == checksum,
            KnowledgeDocument.deleted_at.is_(None),
        )
        .first()
    )


def list_documents(db: Session, owner_username: str, skip: int = 0, limit: int = 50):
    return (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.owner_username == owner_username, KnowledgeDocument.deleted_at.is_(None))
        .order_by(KnowledgeDocument.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_document_status(db: Session, doc_id: int, status: str) -> Optional[KnowledgeDocument]:
    doc = get_document(db, doc_id)
    if doc:
        doc.status = status
        db.commit()
        db.refresh(doc)
    return doc


def delete_document(db: Session, doc_id: int) -> bool:
    """Soft-delete a document (sets deleted_at timestamp)."""
    doc = get_document(db, doc_id)
    if not doc:
        return False
    doc.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return True


def hard_delete_document(db: Session, doc_id: int) -> bool:
    """Permanently delete a document and all related records (admin only)."""
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

def create_chunks_bulk(db: Session, document_id: int, chunks: list[dict]) -> list[KnowledgeChunk]:
    """Insert multiple chunks at once. Each dict: {chunk_index, content, token_count}."""
    rows = []
    for c in chunks:
        row = KnowledgeChunk(
            document_id=document_id,
            chunk_index=c["chunk_index"],
            content=c["content"],
            token_count=c.get("token_count"),
            embedding_model=c.get("embedding_model"),
            vector_id=c.get("vector_id"),
            metadata_json=c.get("metadata_json"),
        )
        rows.append(row)
    db.add_all(rows)
    db.commit()
    for r in rows:
        db.refresh(r)
    return rows


def get_chunks_by_document(db: Session, document_id: int):
    return (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.document_id == document_id)
        .order_by(KnowledgeChunk.chunk_index.asc())
        .all()
    )


def delete_chunks_by_document(db: Session, document_id: int):
    db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document_id).delete(
        synchronize_session=False
    )
    db.commit()


def list_searchable_chunks(
    db: Session,
    owner_username: str,
    document_id: int | None = None,
    *,
    indexed_only: bool = True,
):
    query = (
        db.query(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .filter(KnowledgeDocument.owner_username == owner_username, KnowledgeDocument.deleted_at.is_(None))
    )
    if indexed_only:
        query = query.filter(KnowledgeDocument.status == "indexed")
    if document_id is not None:
        query = query.filter(KnowledgeDocument.id == document_id)
    return query.order_by(KnowledgeDocument.id.asc(), KnowledgeChunk.chunk_index.asc()).all()


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------

def create_ingestion_job(db: Session, document_id: int) -> IngestionJob:
    job = IngestionJob(document_id=document_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_ingestion_job(db: Session, job_id: int) -> Optional[IngestionJob]:
    return db.query(IngestionJob).filter(IngestionJob.id == job_id).first()


def update_ingestion_job_status(
    db: Session, job_id: int, status: str, error_message: Optional[str] = None
) -> Optional[IngestionJob]:
    job = get_ingestion_job(db, job_id)
    if job:
        job.status = status
        job.error_message = error_message
        db.commit()
        db.refresh(job)
    return job


def list_ingestion_jobs(db: Session, document_id: int):
    return (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc())
        .all()
    )


# ---------------------------------------------------------------------------
# Retrieval events
# ---------------------------------------------------------------------------

def create_retrieval_event(
    db: Session,
    username: str,
    query_text: str,
    top_k: int,
    *,
    session_id: int | None = None,
    request_id: str | None = None,
    latency_ms: int | None = None,
    metadata_json: dict | None = None,
) -> RetrievalEvent:
    event = RetrievalEvent(
        username=username,
        session_id=session_id,
        request_id=request_id,
        query_text=query_text,
        top_k=top_k,
        latency_ms=latency_ms,
        metadata_json=metadata_json,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_retrieval_event_by_request_id(db: Session, request_id: str) -> Optional[RetrievalEvent]:
    return (
        db.query(RetrievalEvent)
        .filter(RetrievalEvent.request_id == request_id)
        .order_by(RetrievalEvent.id.desc())
        .first()
    )


def update_retrieval_event_metadata(
    db: Session,
    retrieval_id: int,
    metadata_updates: dict,
) -> Optional[RetrievalEvent]:
    event = db.query(RetrievalEvent).filter(RetrievalEvent.id == retrieval_id).first()
    if not event:
        return None
    merged_metadata = {**(event.metadata_json or {}), **(metadata_updates or {})}
    event.metadata_json = merged_metadata
    db.commit()
    db.refresh(event)
    return event


def list_retrieval_events_by_request_ids(db: Session, request_ids: list[str]) -> list[RetrievalEvent]:
    if not request_ids:
        return []
    return (
        db.query(RetrievalEvent)
        .filter(RetrievalEvent.request_id.in_(request_ids))
        .order_by(RetrievalEvent.id.asc())
        .all()
    )


def replace_answer_citations(
    db: Session,
    request_id: str,
    citations: list[dict],
) -> list[AnswerCitation]:
    db.query(AnswerCitation).filter(AnswerCitation.request_id == request_id).delete(
        synchronize_session=False
    )

    rows: list[AnswerCitation] = []
    for citation in citations:
        row = AnswerCitation(
            request_id=request_id,
            document_id=citation["document_id"],
            chunk_id=citation["chunk_id"],
            rank=citation["rank"],
            score=citation.get("score"),
            quoted_text=citation.get("quoted_text"),
        )
        rows.append(row)

    if rows:
        db.add_all(rows)

    db.commit()

    for row in rows:
        db.refresh(row)

    return rows


def list_answer_citations_by_request_ids(db: Session, request_ids: list[str]) -> list[tuple[AnswerCitation, KnowledgeDocument, KnowledgeChunk]]:
    if not request_ids:
        return []
    return (
        db.query(AnswerCitation, KnowledgeDocument, KnowledgeChunk)
        .join(KnowledgeDocument, AnswerCitation.document_id == KnowledgeDocument.id)
        .join(KnowledgeChunk, AnswerCitation.chunk_id == KnowledgeChunk.id)
        .filter(AnswerCitation.request_id.in_(request_ids))
        .order_by(AnswerCitation.request_id.asc(), AnswerCitation.rank.asc(), AnswerCitation.id.asc())
        .all()
    )


def get_retrieval_analytics(db: Session, *, username: str | None = None, recent_limit: int = 20) -> dict:
    events_query = db.query(RetrievalEvent)
    documents_query = db.query(KnowledgeDocument)
    chunks_query = db.query(KnowledgeChunk)

    if username:
        events_query = events_query.filter(RetrievalEvent.username == username)
        documents_query = documents_query.filter(KnowledgeDocument.owner_username == username)
        chunks_query = chunks_query.join(
            KnowledgeDocument,
            KnowledgeChunk.document_id == KnowledgeDocument.id,
        ).filter(KnowledgeDocument.owner_username == username)

    events = events_query.order_by(RetrievalEvent.created_at.desc(), RetrievalEvent.id.desc()).all()
    recent_events = events[: max(0, recent_limit)]

    request_ids = [event.request_id for event in events if event.request_id]
    citation_counts = Counter()
    if request_ids:
        citation_rows = (
            db.query(AnswerCitation.request_id, func.count(AnswerCitation.id))
            .filter(AnswerCitation.request_id.in_(request_ids))
            .group_by(AnswerCitation.request_id)
            .all()
        )
        citation_counts.update({request_id: int(count) for request_id, count in citation_rows})

    total_events = len(events)
    hit_count = 0
    grounded_count = 0
    weak_count = 0
    fallback_count = 0
    cautious_count = 0
    insufficient_count = 0
    scoped_count = 0
    unscoped_count = 0
    citationless_grounded_count = 0
    total_latency = 0
    latency_samples = 0
    total_returned = 0
    total_citations = 0
    answer_policy_counts: Counter[str] = Counter()
    evidence_strength_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()

    serialized_recent_events: list[dict] = []
    for event in recent_events:
        metadata = event.metadata_json or {}
        returned = int(metadata.get("returned") or 0)
        evidence_strength = metadata.get("evidence_strength") or "none"
        answer_policy = metadata.get("answer_policy") or "unknown"
        fallback_used = bool(metadata.get("fallback_used"))
        document_id = metadata.get("document_id")
        citations_count = int(citation_counts.get(event.request_id or "", 0))
        serialized_recent_events.append(
            {
                "retrieval_id": event.id,
                "request_id": event.request_id,
                "username": event.username,
                "session_id": event.session_id,
                "query_text": event.query_text,
                "returned": returned,
                "top_k": int(event.top_k or 0),
                "latency_ms": int(event.latency_ms or 0) if event.latency_ms is not None else None,
                "strategy": metadata.get("strategy"),
                "evidence_strength": evidence_strength,
                "answer_policy": answer_policy,
                "fallback_used": fallback_used,
                "fallback_reason": metadata.get("fallback_reason"),
                "document_id": int(document_id) if document_id is not None else None,
                "scoped": document_id is not None,
                "rewritten_query": metadata.get("rewritten_query") or event.query_text,
                "citations_count": citations_count,
                "packed_count": int(metadata.get("packed_count") or 0),
                "created_at": event.created_at,
            }
        )

    for event in events:
        metadata = event.metadata_json or {}
        returned = int(metadata.get("returned") or 0)
        evidence_strength = metadata.get("evidence_strength") or "none"
        answer_policy = metadata.get("answer_policy") or "unknown"
        fallback_used = bool(metadata.get("fallback_used"))
        fallback_reason = metadata.get("fallback_reason") or ("used" if fallback_used else "none")
        document_id = metadata.get("document_id")
        citations_count = int(citation_counts.get(event.request_id or "", 0))

        if returned > 0:
            hit_count += 1
        if evidence_strength == "grounded":
            grounded_count += 1
        elif evidence_strength == "weak":
            weak_count += 1
        if fallback_used:
            fallback_count += 1
        if answer_policy == "cautious_general":
            cautious_count += 1
        elif answer_policy == "insufficient_evidence":
            insufficient_count += 1
        if document_id is not None:
            scoped_count += 1
        else:
            unscoped_count += 1
        if answer_policy == "grounded" and citations_count == 0:
            citationless_grounded_count += 1
        if event.latency_ms is not None:
            total_latency += int(event.latency_ms)
            latency_samples += 1
        total_returned += returned
        total_citations += citations_count
        answer_policy_counts.update([answer_policy])
        evidence_strength_counts.update([evidence_strength])
        fallback_reason_counts.update([fallback_reason])

    total_documents = documents_query.count()
    indexed_documents = documents_query.filter(KnowledgeDocument.status == "indexed").count()
    total_chunks = chunks_query.count()

    def _safe_rate(count: int) -> float:
        if total_events == 0:
            return 0.0
        return round(count / total_events, 4)

    summary = {
        "total_events": total_events,
        "hit_rate": _safe_rate(hit_count),
        "grounded_rate": _safe_rate(grounded_count),
        "weak_rate": _safe_rate(weak_count),
        "fallback_rate": _safe_rate(fallback_count),
        "cautious_rate": _safe_rate(cautious_count),
        "insufficient_rate": _safe_rate(insufficient_count),
        "scoped_rate": _safe_rate(scoped_count),
        "avg_latency_ms": round(total_latency / latency_samples, 2) if latency_samples else 0.0,
        "avg_results_returned": round(total_returned / total_events, 2) if total_events else 0.0,
        "avg_citations_per_answer": round(total_citations / total_events, 2) if total_events else 0.0,
        "total_documents": int(total_documents),
        "indexed_documents": int(indexed_documents),
        "total_chunks": int(total_chunks),
        "answer_policy_counts": dict(answer_policy_counts),
        "evidence_strength_counts": dict(evidence_strength_counts),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "citationless_grounded_count": citationless_grounded_count,
        "citationless_grounded_rate": _safe_rate(citationless_grounded_count),
        "scoped_events": scoped_count,
        "unscoped_events": unscoped_count,
    }

    return {
        "username_filter": username,
        "recent_limit": recent_limit,
        "summary": summary,
        "recent_events": serialized_recent_events,
    }


# ---------------------------------------------------------------------------
# Phase 6: Audit Log
# ---------------------------------------------------------------------------

def create_audit_log(
    db: Session,
    actor_username: str,
    action: str,
    *,
    resource_type: str | None = None,
    resource_id: str | None = None,
    request_id: str | None = None,
    detail_json: dict | None = None,
    result_code: int | None = None,
) -> AuditLog:
    """Write an immutable audit log entry. Never raises – best-effort."""
    try:
        entry = AuditLog(
            actor_username=actor_username,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            request_id=request_id,
            detail_json=detail_json,
            result_code=result_code,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry
    except Exception:
        db.rollback()
        raise


def list_audit_logs(
    db: Session,
    *,
    actor_username: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[AuditLog]:
    q = db.query(AuditLog)
    if actor_username:
        q = q.filter(AuditLog.actor_username == actor_username)
    if action:
        q = q.filter(AuditLog.action == action)
    if resource_type:
        q = q.filter(AuditLog.resource_type == resource_type)
    if resource_id:
        q = q.filter(AuditLog.resource_id == resource_id)
    return q.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).offset(skip).limit(limit).all()


# ---------------------------------------------------------------------------
# Phase 6: Cost / usage dashboard
# ---------------------------------------------------------------------------

def get_cost_metrics(db: Session, *, username: str | None = None) -> dict:
    """Aggregate token usage from users table and retrieval latency."""
    from app.models.chat_models import User

    users_q = db.query(User)
    if username:
        users_q = users_q.filter(User.username == username)

    users = users_q.all()

    total_input_tokens = sum(int(u.total_input_tokens_used or 0) for u in users)
    total_output_tokens = sum(int(u.total_output_tokens_used or 0) for u in users)
    total_tokens = sum(int(u.total_token_used or 0) for u in users)

    user_breakdown = [
        {
            "username": u.username,
            "total_tokens": int(u.total_token_used or 0),
            "input_tokens": int(u.total_input_tokens_used or 0),
            "output_tokens": int(u.total_output_tokens_used or 0),
            "max_tokens_per_day": int(u.max_tokens_per_day or 0),
        }
        for u in users
    ]

    # Retrieval latency from retrieval_events
    latency_q = db.query(
        func.count(RetrievalEvent.id),
        func.avg(RetrievalEvent.latency_ms),
        func.min(RetrievalEvent.latency_ms),
        func.max(RetrievalEvent.latency_ms),
    )
    if username:
        latency_q = latency_q.filter(RetrievalEvent.username == username)
    latency_row = latency_q.first()

    total_retrieval_events = int(latency_row[0] or 0)
    avg_latency_ms = round(float(latency_row[1] or 0), 2)
    min_latency_ms = int(latency_row[2] or 0) if latency_row[2] is not None else None
    max_latency_ms = int(latency_row[3] or 0) if latency_row[3] is not None else None

    return {
        "username_filter": username,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_retrieval_events": total_retrieval_events,
        "avg_retrieval_latency_ms": avg_latency_ms,
        "min_retrieval_latency_ms": min_latency_ms,
        "max_retrieval_latency_ms": max_latency_ms,
        "user_breakdown": user_breakdown,
    }
