from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.sql import func

from app.core.database import Base


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    owner_username = Column(String(255), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    source_type = Column(
        Enum("upload", "text", "url", name="knowledge_source_type"),
        nullable=False,
        server_default="text",
    )
    source_uri = Column(String(1024), nullable=True)
    mime_type = Column(String(255), nullable=True)
    status = Column(
        Enum("uploaded", "processing", "indexed", "failed", name="knowledge_document_status"),
        nullable=False,
        server_default="uploaded",
        index=True,
    )
    checksum = Column(String(128), nullable=True, index=True)
    raw_text = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    # Phase 6: soft delete
    deleted_at = Column(DateTime, nullable=True, index=True)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding_model = Column(String(255), nullable=True)
    vector_id = Column(String(255), nullable=True, index=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(
        Enum("queued", "processing", "completed", "failed", name="ingestion_job_status"),
        nullable=False,
        server_default="queued",
        index=True,
    )
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class RetrievalEvent(Base):
    __tablename__ = "retrieval_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(255), nullable=False, index=True)
    session_id = Column(Integer, nullable=True, index=True)
    request_id = Column(String(36), nullable=True, index=True)
    query_text = Column(Text, nullable=False)
    top_k = Column(Integer, nullable=False, default=5)
    latency_ms = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class AnswerCitation(Base):
    __tablename__ = "answer_citations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(36), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey("knowledge_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    score = Column(Float, nullable=True)
    quoted_text = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Phase 6: Audit Log
# ---------------------------------------------------------------------------

class AuditLog(Base):
    """Immutable audit trail for sensitive actions across the application."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # Who performed the action
    actor_username = Column(String(255), nullable=False, index=True)
    # What kind of action: document.upload, document.delete, document.reindex, auth.login, etc.
    action = Column(String(128), nullable=False, index=True)
    # Resource type: document, user, session, chunk
    resource_type = Column(String(64), nullable=True, index=True)
    # Resource primary key (stringified)
    resource_id = Column(String(64), nullable=True, index=True)
    # Request identifier (correlate with retrieval_events)
    request_id = Column(String(36), nullable=True, index=True)
    # Freeform context / diff (JSON)
    detail_json = Column(JSON, nullable=True)
    # HTTP status or result code
    result_code = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

