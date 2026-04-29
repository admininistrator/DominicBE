from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime

from app.core.config import settings


def _allowed_reasoning_efforts_for_model(model_name: str | None) -> set[str]:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return {"low", "medium", "high"}
    if normalized.startswith("gpt-"):
        return {"low", "medium", "high"}
    if normalized == "claude-sonnet-4.6":
        return {"low", "medium", "high"}
    if normalized == "claude-opus-4.6":
        return {"low", "medium", "high", "xhigh"}
    if normalized in {"kc/moonshotai/kimi-k2.6", "kc/qwen/qwen3.6-plus"}:
        return {"instant", "thinking"}
    return set()


class ChatRequest(BaseModel):
    username: Optional[str] = None
    session_id: int
    message: str
    knowledge_document_id: int | None = Field(default=None, ge=1)
    use_web_search: bool = False
    model: str | None = None
    reasoning_effort: str | None = None
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

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        if not normalized:
            return None

        if normalized.startswith("openai/gh/"):
            normalized = normalized.split("openai/gh/", 1)[1]
        elif normalized.startswith("openai/"):
            normalized = normalized.split("openai/", 1)[1]
        if normalized.startswith("gh/"):
            normalized = normalized.split("gh/", 1)[1]

        if normalized not in settings.supported_chat_models:
            raise ValueError(
                "Unsupported chat model. Supported models: " + ", ".join(settings.supported_chat_models)
            )
        return normalized

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip().lower()
        if not normalized:
            return None

        if normalized not in {"low", "medium", "high", "xhigh", "instant", "thinking"}:
            raise ValueError(
                "Unsupported reasoning effort. Supported values: low, medium, high, xhigh, instant, thinking"
            )

        return normalized

    @model_validator(mode="after")
    def _validate_reasoning_effort_for_model(self):
        if not self.reasoning_effort:
            return self

        allowed_efforts = _allowed_reasoning_efforts_for_model(self.model)
        if self.model and self.reasoning_effort not in allowed_efforts:
            raise ValueError(
                "Unsupported reasoning effort '%s' for model '%s'. Supported values: %s"
                % (
                    self.reasoning_effort,
                    self.model,
                    ", ".join(sorted(allowed_efforts)) or "none",
                )
            )
        return self


class CitationSource(BaseModel):
    document_id: int | None = None
    chunk_id: int | None = None
    title: str
    source_type: str = "knowledge"
    score: float | None = None
    rerank_score: float | None = None
    snippet: str
    source_uri: str | None = None
    rank: int | None = None
    url: str | None = None
    domain: str | None = None


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
    web_search_used: bool = False
    web_results_count: int = 0
    web_search_query: str | None = None
    web_latency_ms: int | None = None


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessageDocument(BaseModel):
    id: int | None = None
    title: str
    session_id: int | None = None


class AssistantMessageMeta(BaseModel):
    model: str | None = None
    reasoning_effort: str | None = None
    display_text: str | None = None


class ChatResponse(BaseModel):
    success: bool
    reply: str
    usage: TokenUsage
    request_id: Optional[str] = None
    sources: list[CitationSource] = Field(default_factory=list)
    assistant_meta: AssistantMessageMeta | None = None
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
    documents: list[MessageDocument] = Field(default_factory=list)
    assistant_meta: AssistantMessageMeta | None = None
    input_tokens: int
    output_tokens: int
    created_at: datetime
    request_id: str | None = None
    sources: list[CitationSource] = Field(default_factory=list)
    retrieval: RetrievalMetadata | None = None

