from functools import lru_cache
import os
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus, urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
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
    auth_refresh_token_expire_minutes: int = Field(
        default=60 * 24 * 30,
        alias="AUTH_REFRESH_TOKEN_EXPIRE_MINUTES",
        ge=5,
    )
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_store_backend: Literal["memory", "database"] = Field(
        default="database",
        alias="RATE_LIMIT_STORE_BACKEND",
    )
    rate_limit_trust_proxy_headers: bool = Field(default=True, alias="RATE_LIMIT_TRUST_PROXY_HEADERS")
    rate_limit_cleanup_interval_seconds: int = Field(
        default=60,
        alias="RATE_LIMIT_CLEANUP_INTERVAL_SECONDS",
        ge=5,
    )
    rate_limit_auth_login_requests: int = Field(
        default=5,
        alias="RATE_LIMIT_AUTH_LOGIN_REQUESTS",
        ge=0,
    )
    rate_limit_auth_login_window_seconds: int = Field(
        default=60,
        alias="RATE_LIMIT_AUTH_LOGIN_WINDOW_SECONDS",
        ge=1,
    )
    rate_limit_auth_register_requests: int = Field(
        default=3,
        alias="RATE_LIMIT_AUTH_REGISTER_REQUESTS",
        ge=0,
    )
    rate_limit_auth_register_window_seconds: int = Field(
        default=300,
        alias="RATE_LIMIT_AUTH_REGISTER_WINDOW_SECONDS",
        ge=1,
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_mode(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "dev", "development"}:
                return True
        return value
    rate_limit_auth_reset_password_requests: int = Field(
        default=5,
        alias="RATE_LIMIT_AUTH_RESET_PASSWORD_REQUESTS",
        ge=0,
    )
    rate_limit_auth_reset_password_window_seconds: int = Field(
        default=300,
        alias="RATE_LIMIT_AUTH_RESET_PASSWORD_WINDOW_SECONDS",
        ge=1,
    )
    rate_limit_chat_requests: int = Field(
        default=20,
        alias="RATE_LIMIT_CHAT_REQUESTS",
        ge=0,
    )
    rate_limit_chat_window_seconds: int = Field(
        default=60,
        alias="RATE_LIMIT_CHAT_WINDOW_SECONDS",
        ge=1,
    )
    rate_limit_chat_stream_requests: int = Field(
        default=10,
        alias="RATE_LIMIT_CHAT_STREAM_REQUESTS",
        ge=0,
    )
    rate_limit_chat_stream_window_seconds: int = Field(
        default=60,
        alias="RATE_LIMIT_CHAT_STREAM_WINDOW_SECONDS",
        ge=1,
    )
    auth_password_min_length: int = Field(default=8, alias="AUTH_PASSWORD_MIN_LENGTH", ge=1)
    auth_password_max_length: int = Field(default=16, alias="AUTH_PASSWORD_MAX_LENGTH", ge=8)

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT", ge=1, le=65535)
    web_concurrency: int = Field(default=1, alias="WEB_CONCURRENCY", ge=1)

    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )
    cors_allow_origin_regex_raw: str | None = Field(
        default=None,
        alias="CORS_ALLOW_ORIGIN_REGEX",
    )

    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    db_host: str = Field(default="127.0.0.1", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT", ge=1, le=65535)
    db_user: str = Field(default="dominic", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_name: str = Field(default="chatbot_db", alias="DB_NAME")
    db_ssl: bool = Field(default=False, alias="DB_SSL")
    db_ssl_ca: str | None = Field(default=None, alias="DB_SSL_CA")
    db_charset: str = Field(default="utf8mb4", alias="DB_CHARSET")
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE", ge=0)
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW", ge=0)
    db_pool_recycle: int = Field(default=300, alias="DB_POOL_RECYCLE", ge=1)
    db_pool_timeout: int = Field(default=10, alias="DB_POOL_TIMEOUT", ge=1)
    db_connect_timeout: int = Field(default=10, alias="DB_CONNECT_TIMEOUT", ge=1)
    db_read_timeout: int = Field(default=30, alias="DB_READ_TIMEOUT", ge=1)
    db_write_timeout: int = Field(default=30, alias="DB_WRITE_TIMEOUT", ge=1)
    migration_validation_enabled: bool = Field(default=True, alias="MIGRATION_VALIDATION_ENABLED")
    migration_validation_mode: Literal["warn", "strict"] = Field(
        default="strict",
        alias="MIGRATION_VALIDATION_MODE",
    )

    # OpenAI-compatible LLM provider registry.
    llm_default_provider: str = Field(default="ninerouter", alias="LLM_DEFAULT_PROVIDER")
    llm_default_model: str = Field(default="gpt-5.4", alias="LLM_DEFAULT_MODEL")
    llm_provider_catalog_json: str = Field(default="", alias="LLM_PROVIDER_CATALOG_JSON")
    llm_provider_catalog_file: str = Field(default="", alias="LLM_PROVIDER_CATALOG_FILE")
    llm_context_window: int = Field(default=200000, alias="LLM_CONTEXT_WINDOW", ge=1)
    ninerouter_base_url: str = Field(
        default="http://127.0.0.1:20128/v1",
        alias="NINEROUTER_BASE_URL",
    )
    ninerouter_api_key: str = Field(default="", alias="NINEROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    commandcode_base_url: str = Field(default="", alias="COMMANDCODE_BASE_URL")
    commandcode_api_key: str = Field(default="", alias="COMMANDCODE_API_KEY")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")
    ollama_api_key: str = Field(default="", alias="OLLAMA_API_KEY")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL")
    lmstudio_api_key: str = Field(default="", alias="LMSTUDIO_API_KEY")
    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="", alias="VLLM_API_KEY")
    custom_openai_base_url: str = Field(default="", alias="CUSTOM_OPENAI_BASE_URL")
    custom_openai_api_key: str = Field(default="", alias="CUSTOM_OPENAI_API_KEY")

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
    llm_chat_image_max_count: int = Field(
        default=5,
        alias="LLM_CHAT_IMAGE_MAX_COUNT",
        ge=1,
        le=20,
        description="Maximum number of images allowed per chat message.",
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

    # Prompt caching hook retained for compatibility; current OpenAI-compatible flow is a no-op.
    llm_prompt_caching_enabled: bool = Field(default=True, alias="LLM_PROMPT_CACHING_ENABLED")
    llm_prompt_caching_min_chars: int = Field(
        default=3000,
        alias="LLM_PROMPT_CACHING_MIN_CHARS",
        ge=100,
        description=(
            "Minimum system-prompt character count to apply cache_control. "
            "This is currently retained only for compatibility with older prompt-caching flows."
        ),
    )

    context_window_size: int = Field(default=8, alias="CONTEXT_WINDOW_SIZE", ge=1)
    summary_trigger_messages: int = Field(default=10, alias="SUMMARY_TRIGGER_MESSAGES", ge=1)
    summary_max_tokens: int = Field(default=220, alias="SUMMARY_MAX_TOKENS", ge=32)
    chat_message_max_length: int = Field(
        default=12000,
        alias="CHAT_MESSAGE_MAX_LENGTH",
        ge=1,
        le=200000,
    )
    max_output_tokens: int = Field(default=5000, alias="MAX_OUTPUT_TOKENS", ge=1)
    rolling_window_hours: int = Field(default=2, alias="ROLLING_WINDOW_HOURS", ge=1)
    token_estimate_chars_per_token: int = Field(
        default=4,
        alias="TOKEN_ESTIMATE_CHARS_PER_TOKEN",
        ge=1,
    )

    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="local-hash-v1", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=64, alias="EMBEDDING_DIMENSIONS", ge=1)
    embedding_base_url: str = Field(default="http://localhost:11434", alias="EMBEDDING_BASE_URL")
    embedding_timeout_seconds: float = Field(default=60.0, alias="EMBEDDING_TIMEOUT_SECONDS", ge=1.0)
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE", ge=1, le=256)

    # ── API embedding provider config (Phase 1: Multi-Provider API Embedding) ──
    # Sensitive — never logged, never exposed in errors/health/diagnostics.
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    # API format type: 'openai', 'cohere', 'voyage', 'huggingface', or empty (defaults to openai).
    embedding_api_type: str = Field(default="", alias="EMBEDDING_API_TYPE")
    # Optional API version string (e.g. '2024-02-01' for Azure).
    embedding_api_version: str = Field(default="", alias="EMBEDDING_API_VERSION")
    # JSON string for custom HTTP headers (e.g. '{"X-Custom-Header": "value"}').
    embedding_api_headers: str = Field(default="", alias="EMBEDDING_API_HEADERS")
    ingestion_pipeline: str = Field(default="custom", alias="INGESTION_PIPELINE")
    vector_store_provider: str = Field(default="database", alias="VECTOR_STORE_PROVIDER")
    vector_store_url: str | None = Field(default=None, alias="VECTOR_STORE_URL")
    vector_store_api_key: str | None = Field(default=None, alias="VECTOR_STORE_API_KEY")
    vector_store_collection: str = Field(default="knowledge_chunks", alias="VECTOR_STORE_COLLECTION")
    vector_store_timeout_seconds: float = Field(
        default=10.0,
        alias="VECTOR_STORE_TIMEOUT_SECONDS",
        ge=0.1,
    )
    vector_store_prefer_grpc: bool = Field(default=False, alias="VECTOR_STORE_PREFER_GRPC")
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
    web_search_enabled: bool = Field(default=False, alias="WEB_SEARCH_ENABLED")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS", ge=1, le=10)
    web_search_topic: Literal["general", "news"] = Field(
        default="general",
        alias="WEB_SEARCH_TOPIC",
    )
    web_search_query_planner_enabled: bool = Field(
        default=True,
        alias="WEB_SEARCH_QUERY_PLANNER_ENABLED",
    )
    web_search_query_planner_model: str = Field(
        default="gpt-5.4-mini",
        alias="WEB_SEARCH_QUERY_PLANNER_MODEL",
    )
    web_search_query_planner_max_queries: int = Field(
        default=4,
        alias="WEB_SEARCH_QUERY_PLANNER_MAX_QUERIES",
        ge=2,
        le=6,
    )
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    tavily_base_url: str = Field(default="https://api.tavily.com", alias="TAVILY_BASE_URL")
    rate_limit_knowledge_upload_requests: int = Field(
        default=10,
        alias="RATE_LIMIT_KNOWLEDGE_UPLOAD_REQUESTS",
        ge=0,
    )
    rate_limit_knowledge_upload_window_seconds: int = Field(
        default=300,
        alias="RATE_LIMIT_KNOWLEDGE_UPLOAD_WINDOW_SECONDS",
        ge=1,
    )
    rate_limit_knowledge_ingest_requests: int = Field(
        default=10,
        alias="RATE_LIMIT_KNOWLEDGE_INGEST_REQUESTS",
        ge=0,
    )
    rate_limit_knowledge_ingest_window_seconds: int = Field(
        default=300,
        alias="RATE_LIMIT_KNOWLEDGE_INGEST_WINDOW_SECONDS",
        ge=1,
    )
    tavily_search_depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        alias="TAVILY_SEARCH_DEPTH",
    )
    tavily_timeout_seconds: float = Field(
        default=12.0,
        alias="TAVILY_TIMEOUT_SECONDS",
        ge=1.0,
        le=60.0,
    )

    # Remote MCP (Model Context Protocol) foundation. Disabled by default so
    # existing chat behavior remains unchanged unless explicitly enabled.
    mcp_enabled: bool = Field(default=False, alias="MCP_ENABLED")
    mcp_remote_enabled: bool = Field(default=True, alias="MCP_REMOTE_ENABLED")
    mcp_config_file: str = Field(default="config/mcp_servers.json", alias="MCP_CONFIG_FILE")
    mcp_timeout_seconds: float = Field(default=30.0, alias="MCP_TIMEOUT_SECONDS", ge=0.1)
    mcp_max_retries: int = Field(default=2, alias="MCP_MAX_RETRIES", ge=0, le=10)
    mcp_tool_invocation_enabled: bool = Field(
        default=True,
        alias="MCP_TOOL_INVOCATION_ENABLED",
    )
    mcp_artifact_storage_mode: Literal["inline", "local", "s3"] = Field(
        default="inline",
        alias="MCP_ARTIFACT_STORAGE_MODE",
    )
    mcp_total_budget_seconds: float = Field(default=60.0, alias="MCP_TOTAL_BUDGET_SECONDS", ge=0.1)
    mcp_tool_cache_ttl_seconds: float = Field(default=300.0, alias="MCP_TOOL_CACHE_TTL_SECONDS", ge=0.0)
    mcp_max_artifact_content_bytes: int = Field(default=512000, alias="MCP_MAX_ARTIFACT_CONTENT_BYTES", ge=1024)

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP", ge=0)
    knowledge_max_upload_size_mb: int = Field(
        default=20,
        alias="KNOWLEDGE_MAX_UPLOAD_SIZE_MB",
        ge=1,
        le=200,
    )

    object_storage_provider: str = Field(default="local", alias="OBJECT_STORAGE_PROVIDER")
    object_storage_bucket: str = Field(default="dominic-knowledge", alias="OBJECT_STORAGE_BUCKET")
    object_storage_endpoint: str | None = Field(default=None, alias="OBJECT_STORAGE_ENDPOINT")
    object_storage_access_key: str | None = Field(default=None, alias="OBJECT_STORAGE_ACCESS_KEY")
    object_storage_secret_key: str | None = Field(default=None, alias="OBJECT_STORAGE_SECRET_KEY")
    object_storage_region: str | None = Field(default=None, alias="OBJECT_STORAGE_REGION")
    object_storage_secure: bool = Field(default=True, alias="OBJECT_STORAGE_SECURE")
    object_storage_local_path: str = Field(
        default=str(PROJECT_ROOT / ".storage"),
        alias="OBJECT_STORAGE_LOCAL_PATH",
    )

    # Redis/Celery worker configuration. These settings are the single source
    # of truth for Celery modules; worker code must not read env vars directly.
    celery_enabled: bool = Field(default=False, alias="CELERY_ENABLED")
    celery_broker_url: str = Field(
        default="redis://127.0.0.1:6379/0",
        alias="CELERY_BROKER_URL",
    )
    celery_result_backend: str = Field(
        default="redis://127.0.0.1:6379/1",
        alias="CELERY_RESULT_BACKEND",
    )
    celery_task_always_eager: bool = Field(default=False, alias="CELERY_TASK_ALWAYS_EAGER")
    celery_task_soft_time_limit: int = Field(
        default=300,
        validation_alias=AliasChoices(
            "CELERYD_TASK_SOFT_TIME_LIMIT",
            "CELERY_TASK_SOFT_TIME_LIMIT_SECONDS",
        ),
        ge=1,
    )
    celery_task_time_limit: int = Field(
        default=600,
        validation_alias=AliasChoices(
            "CELERYD_TASK_TIME_LIMIT",
            "CELERY_TASK_TIME_LIMIT_SECONDS",
        ),
        ge=1,
    )

    # Phase 6: retry policy for background indexing
    ingestion_max_retries: int = Field(default=3, alias="INGESTION_MAX_RETRIES", ge=0, le=10)
    ingestion_retry_delay_seconds: float = Field(default=2.0, alias="INGESTION_RETRY_DELAY_SECONDS", ge=0.0)
    ingestion_recovery_enabled: bool = Field(default=True, alias="INGESTION_RECOVERY_ENABLED")
    ingestion_recovery_max_jobs: int = Field(default=20, alias="INGESTION_RECOVERY_MAX_JOBS", ge=0, le=500)
    ingestion_stuck_job_timeout_seconds: int = Field(
        default=900,
        alias="INGESTION_STUCK_JOB_TIMEOUT_SECONDS",
        ge=30,
        le=86400,
    )

    rag_core_mode: Literal["library", "api"] = Field(default="library", alias="RAG_CORE_MODE")
    rag_core_base_url: str = Field(default="http://rag-core:8010", alias="RAG_CORE_BASE_URL")
    rag_core_api_key: str = Field(default="", alias="RAG_CORE_API_KEY")
    rag_core_timeout_seconds: float = Field(default=60.0, alias="RAG_CORE_TIMEOUT_SECONDS", ge=1.0)

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
        if self.migration_validation_mode not in {"warn", "strict"}:
            raise ValueError("MIGRATION_VALIDATION_MODE must be either 'warn' or 'strict'")
        if self.celery_task_time_limit < self.celery_task_soft_time_limit:
            raise ValueError(
                "CELERYD_TASK_TIME_LIMIT must be >= CELERYD_TASK_SOFT_TIME_LIMIT"
            )
        return self

    @property
    def celery_task_soft_time_limit_seconds(self) -> int:
        """Backward-compatibility alias for celery_task_soft_time_limit."""
        return self.celery_task_soft_time_limit

    @property
    def celery_task_time_limit_seconds(self) -> int:
        """Backward-compatibility alias for celery_task_time_limit."""
        return self.celery_task_time_limit

    @property
    def cors_origins(self) -> list[str]:
        origins = [
            item.strip().rstrip("/")
            for item in self.cors_origins_raw.split(",")
            if item.strip()
        ]
        return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def cors_allow_origin_regex(self) -> str | None:
        normalized = (self.cors_allow_origin_regex_raw or "").strip()
        return normalized or None

    def get_llm_runtime_env(self, env_name: str) -> str:
        normalized = (env_name or "").strip().upper()
        if not normalized:
            return ""
        known_values = {
            "NINEROUTER_BASE_URL": self.ninerouter_base_url,
            "NINEROUTER_API_KEY": self.ninerouter_api_key,
            "OPENROUTER_BASE_URL": self.openrouter_base_url,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "COMMANDCODE_BASE_URL": self.commandcode_base_url,
            "COMMANDCODE_API_KEY": self.commandcode_api_key,
            "NVIDIA_BASE_URL": self.nvidia_base_url,
            "NVIDIA_API_KEY": self.nvidia_api_key,
            "GROQ_BASE_URL": self.groq_base_url,
            "GROQ_API_KEY": self.groq_api_key,
            "OLLAMA_BASE_URL": self.ollama_base_url,
            "OLLAMA_API_KEY": self.ollama_api_key,
            "LMSTUDIO_BASE_URL": self.lmstudio_base_url,
            "LMSTUDIO_API_KEY": self.lmstudio_api_key,
            "VLLM_BASE_URL": self.vllm_base_url,
            "VLLM_API_KEY": self.vllm_api_key,
            "CUSTOM_OPENAI_BASE_URL": self.custom_openai_base_url,
            "CUSTOM_OPENAI_API_KEY": self.custom_openai_api_key,
        }
        return str(known_values.get(normalized) or os.environ.get(normalized, "")).strip()

    @property
    def sqlalchemy_database_url(self) -> str:
        if (self.database_url or "").strip():
            return self.database_url.strip()

        encoded_password = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{self.db_user}:{encoded_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sqlalchemy_dialect_name(self) -> str:
        parsed = urlparse(self.sqlalchemy_database_url)
        return (parsed.scheme or "").split("+", 1)[0].lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
