from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="Dominic Backend", alias="APP_NAME")
    environment: Literal["local", "dev", "staging", "prod"] = Field(
        default="local",
        alias="ENVIRONMENT",
    )
    debug: bool = Field(default=False, alias="DEBUG")
    enable_debug_env: bool = Field(default=False, alias="ENABLE_DEBUG_ENV")
    auth_secret_key: str = Field(default="change-this-in-production", alias="AUTH_SECRET_KEY")
    auth_algorithm: str = Field(default="HS256", alias="AUTH_ALGORITHM")
    auth_access_token_expire_minutes: int = Field(
        default=60 * 24 * 7,
        alias="AUTH_ACCESS_TOKEN_EXPIRE_MINUTES",
        ge=5,
    )
    auth_password_min_length: int = Field(default=8, alias="AUTH_PASSWORD_MIN_LENGTH", ge=1)
    auth_password_max_length: int = Field(default=64, alias="AUTH_PASSWORD_MAX_LENGTH", ge=8)

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT", ge=1, le=65535)
    web_concurrency: int = Field(default=1, alias="WEB_CONCURRENCY", ge=1)

    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )

    db_host: str = Field(default="127.0.0.1", alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT", ge=1, le=65535)
    db_user: str = Field(default="dominic", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_name: str = Field(default="chatbot_db", alias="DB_NAME")
    db_ssl: bool = Field(default=False, alias="DB_SSL")
    db_ssl_ca: str | None = Field(default=None, alias="DB_SSL_CA")
    db_charset: str = Field(default="utf8mb4", alias="DB_CHARSET")
    db_pool_recycle: int = Field(default=300, alias="DB_POOL_RECYCLE", ge=1)
    db_pool_timeout: int = Field(default=10, alias="DB_POOL_TIMEOUT", ge=1)
    db_connect_timeout: int = Field(default=10, alias="DB_CONNECT_TIMEOUT", ge=1)
    db_read_timeout: int = Field(default=30, alias="DB_READ_TIMEOUT", ge=1)
    db_write_timeout: int = Field(default=30, alias="DB_WRITE_TIMEOUT", ge=1)

    # ── Legacy Anthropic-specific settings (kept for backward compatibility) ────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-latest",
        alias="ANTHROPIC_MODEL",
    )
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")
    anthropic_force_ipv4: bool = Field(default=False, alias="ANTHROPIC_FORCE_IPV4")

    # ── LiteLLM multi-provider settings ──────────────────────────────────────
    # Full LiteLLM model string, e.g.:
    #   "anthropic/claude-3-5-haiku-latest"
    #   "openai/gpt-4o"
    #   "gemini/gemini-1.5-pro"
    #   "azure/gpt-4o"
    #   "ollama/llama3"
    # If blank, falls back to anthropic_model with "anthropic/" prefix.
    llm_model: str = Field(default="", alias="LLM_MODEL")
    # Separate vision model (optional – defaults to llm_model)
    llm_vision_model: str = Field(default="", alias="LLM_VISION_MODEL")

    # Additional provider API keys (only needed when using that provider)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Vision / image features
    llm_vision_enabled: bool = Field(default=True, alias="LLM_VISION_ENABLED")
    llm_image_captioning_enabled: bool = Field(
        default=False,
        alias="LLM_IMAGE_CAPTIONING_ENABLED",
        description=(
            "When True, images embedded in uploaded documents (PDF/DOCX/PPTX) "
            "are described by the vision model and injected into the RAG chunks. "
            "Incurs extra LLM token cost per image."
        ),
    )
    llm_image_caption_max_tokens: int = Field(
        default=256,
        alias="LLM_IMAGE_CAPTION_MAX_TOKENS",
        ge=32,
        le=2048,
    )
    llm_chat_image_max_size_mb: float = Field(
        default=5.0,
        alias="LLM_CHAT_IMAGE_MAX_SIZE_MB",
        ge=0.1,
        le=20.0,
    )

    # Image preprocessing (resize + OCR)
    llm_image_resize_enabled: bool = Field(default=True, alias="LLM_IMAGE_RESIZE_ENABLED")
    llm_image_max_dimension: int = Field(
        default=1568,
        alias="LLM_IMAGE_MAX_DIMENSION",
        ge=256,
        le=4096,
        description="Longest side in pixels after resize. 1568 = Anthropic recommended optimum.",
    )
    llm_image_ocr_enabled: bool = Field(default=True, alias="LLM_IMAGE_OCR_ENABLED")
    llm_image_ocr_confidence_threshold: float = Field(
        default=0.55,
        alias="LLM_IMAGE_OCR_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description=(
            "Min OCR confidence (0–1) to classify image as text-heavy and use "
            "extracted text instead of vision model. 0.55 is a balanced default."
        ),
    )

    # Prompt caching (Anthropic only – transparent no-op for other providers)
    llm_prompt_caching_enabled: bool = Field(default=True, alias="LLM_PROMPT_CACHING_ENABLED")
    llm_prompt_caching_min_chars: int = Field(
        default=3000,
        alias="LLM_PROMPT_CACHING_MIN_CHARS",
        ge=100,
        description=(
            "Minimum system-prompt character count to apply cache_control. "
            "Anthropic requires ≥1024 tokens (~4000 chars) for haiku, "
            "≥2048 tokens for other models.  3000 chars is a safe default."
        ),
    )

    context_window_size: int = Field(default=8, alias="CONTEXT_WINDOW_SIZE", ge=1)
    summary_trigger_messages: int = Field(default=10, alias="SUMMARY_TRIGGER_MESSAGES", ge=1)
    summary_max_tokens: int = Field(default=220, alias="SUMMARY_MAX_TOKENS", ge=32)
    max_output_tokens: int = Field(default=5000, alias="MAX_OUTPUT_TOKENS", ge=1)
    rolling_window_hours: int = Field(default=2, alias="ROLLING_WINDOW_HOURS", ge=1)
    token_estimate_chars_per_token: int = Field(
        default=4,
        alias="TOKEN_ESTIMATE_CHARS_PER_TOKEN",
        ge=1,
    )

    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="local-hash-v1", alias="EMBEDDING_MODEL")
    vector_store_provider: str = Field(default="database", alias="VECTOR_STORE_PROVIDER")
    vector_store_url: str | None = Field(default=None, alias="VECTOR_STORE_URL")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K", ge=1)
    retrieval_min_score: float = Field(default=0.15, alias="RETRIEVAL_MIN_SCORE", ge=0.0, le=1.0)
    retrieval_min_lexical_score: float = Field(
        default=0.1,
        alias="RETRIEVAL_MIN_LEXICAL_SCORE",
        ge=0.0,
        le=1.0,
    )
    retrieval_hybrid_semantic_weight: float = Field(
        default=0.4,
        alias="RETRIEVAL_HYBRID_SEMANTIC_WEIGHT",
        ge=0.0,
        le=1.0,
    )
    retrieval_hybrid_lexical_weight: float = Field(
        default=0.6,
        alias="RETRIEVAL_HYBRID_LEXICAL_WEIGHT",
        ge=0.0,
        le=1.0,
    )
    retrieval_enable_query_expansion: bool = Field(
        default=True,
        alias="RETRIEVAL_ENABLE_QUERY_EXPANSION",
    )
    retrieval_max_rerank_candidates: int = Field(
        default=12,
        alias="RETRIEVAL_MAX_RERANK_CANDIDATES",
        ge=1,
        le=100,
    )
    retrieval_rerank_title_weight: float = Field(
        default=0.15,
        alias="RETRIEVAL_RERANK_TITLE_WEIGHT",
        ge=0.0,
        le=1.0,
    )
    retrieval_rerank_position_weight: float = Field(
        default=0.1,
        alias="RETRIEVAL_RERANK_POSITION_WEIGHT",
        ge=0.0,
        le=1.0,
    )
    retrieval_low_confidence_score: float = Field(
        default=0.2,
        alias="RETRIEVAL_LOW_CONFIDENCE_SCORE",
        ge=0.0,
        le=1.0,
    )
    retrieval_strict_grounding_for_scoped_docs: bool = Field(
        default=True,
        alias="RETRIEVAL_STRICT_GROUNDING_FOR_SCOPED_DOCS",
    )
    answer_guardrails_enabled: bool = Field(
        default=True,
        alias="ANSWER_GUARDRAILS_ENABLED",
    )
    answer_guardrails_allow_weak_citations: bool = Field(
        default=False,
        alias="ANSWER_GUARDRAILS_ALLOW_WEAK_CITATIONS",
    )
    retrieval_max_context_tokens: int = Field(
        default=4000,
        alias="RETRIEVAL_MAX_CONTEXT_TOKENS",
        ge=64,
    )
    retrieval_max_context_chunks: int = Field(
        default=6,
        alias="RETRIEVAL_MAX_CONTEXT_CHUNKS",
        ge=1,
        le=20,
    )
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP", ge=0)
    knowledge_max_upload_size_mb: int = Field(
        default=20,
        alias="KNOWLEDGE_MAX_UPLOAD_SIZE_MB",
        ge=1,
        le=200,
    )

    # Phase 6: retry policy for background indexing
    ingestion_max_retries: int = Field(default=3, alias="INGESTION_MAX_RETRIES", ge=0, le=10)
    ingestion_retry_delay_seconds: float = Field(default=2.0, alias="INGESTION_RETRY_DELAY_SECONDS", ge=0.0)

    # Phase 6: audit log
    audit_log_enabled: bool = Field(default=True, alias="AUDIT_LOG_ENABLED")

    @model_validator(mode="after")
    def validate_auth_password_range(self):
        if self.auth_password_max_length < self.auth_password_min_length:
            raise ValueError("AUTH_PASSWORD_MAX_LENGTH must be >= AUTH_PASSWORD_MIN_LENGTH")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.environment != "local" and self.auth_secret_key == "change-this-in-production":
            raise ValueError("AUTH_SECRET_KEY must be overridden outside local environment")
        total_retrieval_weight = self.retrieval_hybrid_semantic_weight + self.retrieval_hybrid_lexical_weight
        if total_retrieval_weight <= 0:
            raise ValueError("Hybrid retrieval weights must sum to a positive value")
        return self

    @property
    def cors_origins(self) -> list[str]:
        origins = [
            item.strip().rstrip("/")
            for item in self.cors_origins_raw.split(",")
            if item.strip()
        ]
        return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def sqlalchemy_database_url(self) -> str:
        encoded_password = quote_plus(self.db_password)
        return (
            f"mysql+pymysql://{self.db_user}:{encoded_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?charset={self.db_charset}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

