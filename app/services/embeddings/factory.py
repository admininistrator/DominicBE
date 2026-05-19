"""Embedding provider factory.

Centralizes provider selection so that knowledge_service and retrieval_service
never hard-code provider classes or model names.

Usage:
    from app.services.embeddings.factory import get_embedding_provider

    provider = get_embedding_provider()
    result = provider.embed_texts(["hello world"])

The factory reads EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
EMBEDDING_BASE_URL, EMBEDDING_TIMEOUT_SECONDS, and EMBEDDING_BATCH_SIZE from
settings. Unknown provider values raise a clear configuration error.

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import json
import logging

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider, EmbeddingProviderError

logger = logging.getLogger(__name__)


def get_embedding_provider(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    batch_size: int | None = None,
    api_key: str | None = None,
    api_type: str | None = None,
    api_version: str | None = None,
    custom_headers: dict | None = None,
) -> EmbeddingProvider:
    """Return the configured embedding provider instance.

    All parameters are optional and fall back to settings when not supplied.
    This allows callers to override for testing without patching globals.

    Args:
        provider_name: Override EMBEDDING_PROVIDER (e.g. ``'local'``,
            ``'ollama'``, ``'api'``).
        model: Override EMBEDDING_MODEL.
        dimensions: Override EMBEDDING_DIMENSIONS.
        base_url: Override EMBEDDING_BASE_URL.
        timeout_seconds: Override EMBEDDING_TIMEOUT_SECONDS.
        batch_size: Override EMBEDDING_BATCH_SIZE.
        api_key: Override EMBEDDING_API_KEY (used when provider is ``'api'``).
        api_type: Override EMBEDDING_API_TYPE (used when provider is ``'api'``).
        api_version: Override EMBEDDING_API_VERSION (used when provider is ``'api'``).
        custom_headers: Override EMBEDDING_API_HEADERS parsed as dict (used when
            provider is ``'api'``).

    Returns:
        An EmbeddingProvider instance ready to call ``embed_texts`` / ``embed_query``.

    Raises:
        EmbeddingProviderError: If provider_name is not a known provider
            (``'local'``, ``'ollama'``, ``'api'``).
    """
    resolved_provider = (provider_name or settings.embedding_provider or "local").strip().lower()
    resolved_model = (model or settings.embedding_model or "local-hash-v1").strip()
    resolved_dimensions = dimensions if dimensions is not None else settings.embedding_dimensions

    if resolved_provider == "local":
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        return LocalHashProvider(
            model=resolved_model,
            dimensions=resolved_dimensions,
        )

    if resolved_provider == "ollama":
        from app.services.embeddings.ollama_provider import OllamaProvider

        resolved_base_url = (base_url or settings.embedding_base_url or "http://localhost:11434").strip()
        resolved_timeout = timeout_seconds if timeout_seconds is not None else settings.embedding_timeout_seconds
        resolved_batch = batch_size if batch_size is not None else settings.embedding_batch_size

        return OllamaProvider(
            model=resolved_model,
            base_url=resolved_base_url,
            timeout_seconds=resolved_timeout,
            batch_size=resolved_batch,
            expected_dimensions=resolved_dimensions,
        )

    if resolved_provider == "api":
        from app.services.embeddings.generic_api_provider import GenericAPIProvider

        resolved_api_key = api_key if api_key is not None else settings.embedding_api_key
        resolved_api_type = api_type if api_type is not None else settings.embedding_api_type
        resolved_api_version = api_version if api_version is not None else settings.embedding_api_version
        resolved_headers = custom_headers if custom_headers is not None else _parse_embedding_api_headers()
        resolved_base_url = (base_url or settings.embedding_base_url or "").strip()
        resolved_timeout = timeout_seconds if timeout_seconds is not None else settings.embedding_timeout_seconds
        resolved_batch = batch_size if batch_size is not None else settings.embedding_batch_size

        return GenericAPIProvider(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_type=resolved_api_type,
            timeout_seconds=resolved_timeout,
            batch_size=resolved_batch,
            expected_dimensions=resolved_dimensions,
            api_version=resolved_api_version,
            custom_headers=resolved_headers,
        )

    raise EmbeddingProviderError(
        (
            f"Unknown EMBEDDING_PROVIDER={resolved_provider!r}. "
            "Supported values: 'local', 'ollama', 'api'."
        ),
        provider=resolved_provider,
        model=resolved_model,
        category="configuration_error",
    )


def _parse_embedding_api_headers() -> dict:
    """Parse ``EMBEDDING_API_HEADERS`` from settings as a JSON dict.

    Returns:
        Parsed headers dict, or empty dict on parse failure.
    """
    raw = (settings.embedding_api_headers or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
        logger.warning("EMBEDDING_API_HEADERS is not a JSON object, ignoring.")
        return {}
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse EMBEDDING_API_HEADERS: %s", exc)
        return {}
