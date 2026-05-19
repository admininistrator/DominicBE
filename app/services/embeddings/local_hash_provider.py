"""Local deterministic hash embedding provider (local-hash-v1).

This provider replicates the exact behavior of the original
``compute_text_embedding()`` in knowledge_service.py so that existing
ingested chunks remain compatible after the provider abstraction is
introduced.

Backward-compatibility guarantee:
- Same deterministic output for the same input text.
- Same default dimension (64, from EMBEDDING_DIMENSIONS).
- Provider name: 'local', version: 'local-hash-v1'.

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import hashlib
import re
import time

from app.services.embeddings.base import (
    EmbeddingMeta,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)

_PROVIDER_NAME = "local"
_VERSION = "local-hash-v1"


def _normalize_text(text: str) -> str:
    """Normalize whitespace for embedding (mirrors knowledge_service logic)."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    paragraphs = []
    for block in re.split(r"\n\s*\n+", raw):
        normalized = re.sub(r"[ \t]+", " ", block).strip()
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def _hash_embed(text: str, dimensions: int) -> list[float]:
    """Deterministic local hash embedding – identical to original compute_text_embedding()."""
    normalized = _normalize_text(text).lower()
    if not normalized:
        return [0.0] * dimensions

    vector = [0.0] * dimensions
    for token in re.findall(r"\w+", normalized, flags=re.UNICODE):
        token_hash = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(token_hash[:2], "big") % dimensions
        sign = 1.0 if token_hash[2] % 2 == 0 else -1.0
        weight = 1.0 + ((token_hash[3] % 5) * 0.1)
        vector[bucket] += sign * weight

    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return [0.0] * dimensions
    return [round(value / magnitude, 6) for value in vector]


class LocalHashProvider:
    """Deterministic local hash embedding provider.

    Uses the same algorithm as the original ``compute_text_embedding()``
    so existing chunk vectors remain valid after the provider refactor.
    """

    def __init__(self, *, model: str = _VERSION, dimensions: int = 64) -> None:
        self._model = model or _VERSION
        self._dimensions = dimensions
        self._meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=self._dimensions,
            version=_VERSION,
        )

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Embed a batch of texts using the local hash algorithm.

        Args:
            texts: Non-empty list of text strings.

        Returns:
            EmbedResult with one vector per input text.

        Raises:
            EmbeddingProviderError: If texts is empty.
        """
        if not texts:
            raise EmbeddingProviderError(
                "embed_texts requires at least one text",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="invalid_input",
            )
        started = time.perf_counter()
        vectors = [_hash_embed(t, self._dimensions) for t in texts]
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=self._dimensions,
            version=_VERSION,
            extra={"latency_ms": latency_ms, "batch_size": len(texts)},
        )
        return EmbedResult(vectors=vectors, meta=meta)

    def embed_query(self, query: str) -> QueryEmbedResult:
        """Embed a single query string.

        Args:
            query: Non-empty query text.

        Returns:
            QueryEmbedResult with a single vector.

        Raises:
            EmbeddingProviderError: If query is empty.
        """
        if not (query or "").strip():
            raise EmbeddingProviderError(
                "embed_query requires a non-empty query",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="invalid_input",
            )
        started = time.perf_counter()
        vector = _hash_embed(query, self._dimensions)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=self._dimensions,
            version=_VERSION,
            extra={"latency_ms": latency_ms},
        )
        return QueryEmbedResult(vector=vector, meta=meta)

    @property
    def meta(self) -> EmbeddingMeta:
        return self._meta
