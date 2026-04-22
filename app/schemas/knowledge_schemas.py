from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMBaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class KnowledgeDocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    source_type: str = Field(default="text")
    source_uri: str | None = None
    mime_type: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] | None = None


class KnowledgeDocumentResponse(ORMBaseSchema):
    id: int
    owner_username: str
    title: str
    source_type: str
    source_uri: str | None = None
    mime_type: str | None = None
    status: str
    checksum: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class KnowledgeChunkResponse(ORMBaseSchema):
    id: int
    document_id: int
    chunk_index: int
    content: str
    token_count: int | None = None
    embedding_model: str | None = None
    vector_id: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime


class IngestionJobResponse(ORMBaseSchema):
    id: int
    document_id: int
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class IngestionResult(BaseModel):
    document_id: int
    job_id: int
    status: str
    chunks_count: int = 0
    checksum: str | None = None
    attempts: int | None = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_id: int | None = Field(default=None, ge=1)


class KnowledgeSearchResult(BaseModel):
    document_id: int
    chunk_id: int
    chunk_index: int
    title: str
    source_type: str
    source_uri: str | None = None
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    token_estimate: int | None = None
    snippet: str
    vector_id: str | None = None
    embedding_model: str | None = None


class KnowledgeSearchResponse(BaseModel):
    query: str
    top_k: int
    returned: int
    retrieval_id: int | None = None
    latency_ms: int | None = None
    candidate_count: int | None = None
    document_id: int | None = None
    strategy: str | None = None
    original_query: str | None = None
    rewritten_query: str | None = None
    query_expansions: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None
    evidence_strength: str | None = None
    results: list[KnowledgeSearchResult]


class CitationSource(BaseModel):
    document_id: int
    chunk_id: int
    title: str
    score: float | None = None
    snippet: str


class RetrievalAnalyticsSummary(BaseModel):
    total_events: int
    hit_rate: float
    grounded_rate: float
    weak_rate: float
    fallback_rate: float
    cautious_rate: float
    insufficient_rate: float
    scoped_rate: float
    avg_latency_ms: float
    avg_results_returned: float
    avg_citations_per_answer: float
    total_documents: int
    indexed_documents: int
    total_chunks: int
    answer_policy_counts: dict[str, int] = Field(default_factory=dict)
    evidence_strength_counts: dict[str, int] = Field(default_factory=dict)
    fallback_reason_counts: dict[str, int] = Field(default_factory=dict)
    citationless_grounded_count: int = 0
    citationless_grounded_rate: float = 0.0
    scoped_events: int = 0
    unscoped_events: int = 0


class RetrievalAnalyticsEvent(BaseModel):
    retrieval_id: int
    request_id: str | None = None
    username: str
    session_id: int | None = None
    query_text: str
    returned: int
    top_k: int
    latency_ms: int | None = None
    strategy: str | None = None
    evidence_strength: str | None = None
    answer_policy: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    document_id: int | None = None
    scoped: bool = False
    rewritten_query: str | None = None
    citations_count: int = 0
    packed_count: int = 0
    created_at: datetime


class RetrievalAnalyticsResponse(BaseModel):
    username_filter: str | None = None
    recent_limit: int
    summary: RetrievalAnalyticsSummary
    recent_events: list[RetrievalAnalyticsEvent]


# ---------------------------------------------------------------------------
# Phase 6: Audit Log
# ---------------------------------------------------------------------------

class AuditLogResponse(BaseModel):
    id: int
    actor_username: str
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    request_id: str | None = None
    detail_json: dict[str, Any] | None = None
    result_code: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Phase 6: Cost / usage dashboard
# ---------------------------------------------------------------------------

class UserTokenBreakdown(BaseModel):
    username: str
    total_tokens: int
    input_tokens: int
    output_tokens: int
    max_tokens_per_day: int


class CostMetricsResponse(BaseModel):
    username_filter: str | None = None
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_retrieval_events: int
    avg_retrieval_latency_ms: float
    min_retrieval_latency_ms: int | None = None
    max_retrieval_latency_ms: int | None = None
    user_breakdown: list[UserTokenBreakdown]


