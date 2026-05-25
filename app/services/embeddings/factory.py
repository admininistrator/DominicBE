"""Embedding provider factory.

WARNING: This file now delegates to ``rag_core.embeddings.factory``.
The factory reads default values from ``app.core.config.settings`` and passes
them to the rag-core factory.  All existing import paths continue to work.

Do NOT edit provider selection logic here — modify ``rag_core.embeddings.factory`` instead.
"""
from __future__ import annotations

import json
import logging

from app.core.config import settings
from rag_core.embeddings.base import EmbeddingProvider, EmbeddingProviderError
from rag_core.embeddings.factory import get_embedding_provider as _rag_core_get_embedding_provider

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
    This delegates to ``rag_core.embeddings.factory.get_embedding_provider``
    with defaults from DominicBE's ``settings``.

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
    resolved_headers = custom_headers
    if resolved_headers is None:
        resolved_headers = _parse_embedding_api_headers()

    return _rag_core_get_embedding_provider(
        provider_name=provider_name,
        model=model,
        dimensions=dimensions,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        api_key=api_key,
        api_type=api_type,
        api_version=api_version,
        custom_headers=resolved_headers,
        default_provider=(settings.embedding_provider or "local"),
        default_model=(settings.embedding_model or "local-hash-v1"),
        default_dimensions=settings.embedding_dimensions,
        default_base_url=(settings.embedding_base_url or "http://localhost:11434"),
        default_timeout_seconds=settings.embedding_timeout_seconds,
        default_batch_size=settings.embedding_batch_size,
        default_api_key=settings.embedding_api_key,
        default_api_type=settings.embedding_api_type,
        default_api_version=settings.embedding_api_version,
        default_custom_headers=resolved_headers,
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
