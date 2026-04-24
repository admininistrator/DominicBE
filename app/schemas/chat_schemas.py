from typing import Optional

from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ChatRequest(BaseModel):
    username: Optional[str] = None
    session_id: int
    message: str
    knowledge_document_id: int | None = Field(default=None, ge=1)
    # Vision: list of base64-encoded image strings or data-URIs.
    # Only used when LLM_VISION_ENABLED=true and the model supports vision.
    images: list[str] = Field(default_factory=list)
    # Parallel MIME types for each image (e.g. "image/jpeg", "image/png").
    # Defaults to "image/jpeg" when omitted.
    image_media_types: list[str] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _validate_image_count(cls, v: list[str]) -> list[str]:
        if len(v) > 10:
            raise ValueError("Maximum 10 images per message.")
        return v


class CitationSource(BaseModel):
    document_id: int
    chunk_id: int
    title: str
    score: float | None = None
    rerank_score: float | None = None
    snippet: str
    source_uri: str | None = None
    rank: int | None = None


class RetrievalMetadata(BaseModel):
    used: bool
    top_k: int
    returned: int
    retrieval_id: int | None = None
    latency_ms: int | None = None
    document_id: int | None = None
    strategy: str | None = None
    original_query: str | None = None
    rewritten_query: str | None = None
    query_expansions: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None
    evidence_strength: str | None = None
    answer_policy: str | None = None
    packed_count: int = 0
    packed_token_estimate: int = 0


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    success: bool
    reply: str
    usage: TokenUsage
    request_id: Optional[str] = None
    sources: list[CitationSource] = Field(default_factory=list)
    retrieval: RetrievalMetadata | None = None


class UsageResponse(BaseModel):
    username: str
    max_tokens_per_day: int
    total_token_used: int
    total_input_tokens_used: int
    total_output_tokens_used: int
    lifetime_total_token_used: int
    lifetime_total_input_tokens_used: int
    lifetime_total_output_tokens_used: int
    rolling_window_hours: int
    rolling_total_token_used: int
    rolling_input_tokens_used: int
    rolling_output_tokens_used: int


class SessionCreateRequest(BaseModel):
    username: Optional[str] = None
    title: Optional[str] = None


class SessionRenameRequest(BaseModel):
    title: str


class SessionResponse(BaseModel):
    id: int
    username: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    images: list[str] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    created_at: datetime
    request_id: str | None = None
    sources: list[CitationSource] = Field(default_factory=list)
    retrieval: RetrievalMetadata | None = None

