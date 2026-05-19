"""Ollama embedding provider using POST /api/embed.

This provider calls the Ollama REST API to produce real semantic embeddings.
It is disabled by default (EMBEDDING_PROVIDER defaults to 'local').
To enable: set EMBEDDING_PROVIDER=ollama in your environment.

Design constraints:
- Uses httpx (already in requirements.txt) – no new HTTP dependency.
- Model name comes from EMBEDDING_MODEL config; never hard-coded here.
- Logs latency, provider, model, dimensions, batch size, and failure category.
- Never logs raw document text.
- Retries only transient network failures and HTTP 5xx responses.
- Does not retry validation failures (missing embeddings, mismatched count,
  non-numeric values, inconsistent dimensions).
- Fails the entire batch on error rather than silently falling back.

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import logging
import math
import time
from urllib.parse import urljoin

from app.services.embeddings.base import (
    EmbeddingDimensionMismatchError,
    EmbeddingMeta,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "ollama"
_VERSION = "ollama-embed-v1"
_EMBED_PATH = "/api/embed"

# Transient HTTP status codes that are safe to retry
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.0


def _is_finite(value: object) -> bool:
    try:
        f = float(value)  # type: ignore[arg-type]
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _validate_response(
    data: dict,
    *,
    expected_count: int,
    provider: str,
    model: str,
    url: str,
) -> list[list[float]]:
    """Validate Ollama /api/embed response shape.

    Raises:
        EmbeddingProviderError: For missing or malformed embeddings field.
        EmbeddingDimensionMismatchError: For inconsistent vector dimensions.
    """
    raw = data.get("embeddings")
    if not isinstance(raw, list) or not raw:
        raise EmbeddingProviderError(
            f"Ollama response missing 'embeddings' list (url={url}, model={model})",
            provider=provider,
            model=model,
            category="invalid_response",
        )

    if len(raw) != expected_count:
        raise EmbeddingProviderError(
            (
                f"Ollama returned {len(raw)} embeddings but expected {expected_count} "
                f"(url={url}, model={model})"
            ),
            provider=provider,
            model=model,
            category="count_mismatch",
        )

    vectors: list[list[float]] = []
    first_dim: int | None = None

    for idx, vec in enumerate(raw):
        if not isinstance(vec, list) or not vec:
            raise EmbeddingProviderError(
                f"Ollama embedding[{idx}] is empty or not a list (url={url}, model={model})",
                provider=provider,
                model=model,
                category="invalid_response",
            )
        if not all(_is_finite(v) for v in vec):
            raise EmbeddingProviderError(
                f"Ollama embedding[{idx}] contains non-finite values (url={url}, model={model})",
                provider=provider,
                model=model,
                category="non_numeric_values",
            )
        dim = len(vec)
        if first_dim is None:
            first_dim = dim
        elif dim != first_dim:
            raise EmbeddingDimensionMismatchError(
                (
                    f"Ollama embedding[{idx}] has dimension {dim} but embedding[0] "
                    f"has dimension {first_dim} (url={url}, model={model})"
                ),
                provider=provider,
                model=model,
                expected=first_dim,
                actual=dim,
            )
        vectors.append([float(v) for v in vec])

    return vectors


class OllamaProvider:
    """Ollama embedding provider.

    Calls POST /api/embed on the configured Ollama base URL.
    Only active when EMBEDDING_PROVIDER=ollama.

    Args:
        model: Model name from config (e.g. 'qwen3-embedding:0.6b').
        base_url: Ollama base URL (e.g. 'http://localhost:11434').
        timeout_seconds: HTTP timeout for each request.
        batch_size: Maximum texts per single /api/embed call.
        expected_dimensions: If > 0, validate that returned vectors match.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        batch_size: int = 16,
        expected_dimensions: int = 0,
    ) -> None:
        self._model = model
        self._base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._timeout = timeout_seconds
        self._batch_size = max(1, batch_size)
        self._expected_dimensions = expected_dimensions
        self._embed_url = self._base_url + _EMBED_PATH
        # Dimensions are discovered on first successful call when not pre-configured
        self._discovered_dimensions: int = expected_dimensions

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Embed a batch of texts via Ollama POST /api/embed.

        Splits into sub-batches of at most batch_size texts.

        Args:
            texts: Non-empty ordered list of text strings.

        Returns:
            EmbedResult with vectors in the same order as input texts.

        Raises:
            EmbeddingProviderError: On connection error, timeout, HTTP error,
                or invalid response shape.
        """
        if not texts:
            raise EmbeddingProviderError(
                "embed_texts requires at least one text",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="invalid_input",
            )

        started = time.perf_counter()
        all_vectors: list[list[float]] = []

        # Process in bounded batches
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            batch_vectors = self._call_api(batch)
            all_vectors.extend(batch_vectors)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        dims = len(all_vectors[0]) if all_vectors else self._discovered_dimensions

        logger.info(
            "OllamaProvider.embed_texts: provider=%s model=%s dims=%d batch=%d latency_ms=%.1f",
            _PROVIDER_NAME,
            self._model,
            dims,
            len(texts),
            latency_ms,
        )

        meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=dims,
            version=_VERSION,
            extra={"latency_ms": latency_ms, "batch_size": len(texts)},
        )
        return EmbedResult(vectors=all_vectors, meta=meta)

    def embed_query(self, query: str) -> QueryEmbedResult:
        """Embed a single query string via Ollama.

        Args:
            query: Non-empty query text.

        Returns:
            QueryEmbedResult with a single vector.

        Raises:
            EmbeddingProviderError: On any provider-level failure.
        """
        if not (query or "").strip():
            raise EmbeddingProviderError(
                "embed_query requires a non-empty query",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="invalid_input",
            )

        started = time.perf_counter()
        vectors = self._call_api([query])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        vector = vectors[0]
        dims = len(vector)

        logger.info(
            "OllamaProvider.embed_query: provider=%s model=%s dims=%d latency_ms=%.1f",
            _PROVIDER_NAME,
            self._model,
            dims,
            latency_ms,
        )

        meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=dims,
            version=_VERSION,
            extra={"latency_ms": latency_ms},
        )
        return QueryEmbedResult(vector=vector, meta=meta)

    @property
    def meta(self) -> EmbeddingMeta:
        return EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=self._discovered_dimensions,
            version=_VERSION,
        )

    # ------------------------------------------------------------------
    # Internal HTTP call with retry
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """POST /api/embed with retry on transient failures.

        Args:
            texts: Batch of texts (already bounded by batch_size).

        Returns:
            List of float vectors in input order.

        Raises:
            EmbeddingProviderError: On non-retryable failure or exhausted retries.
        """
        try:
            import httpx
        except ImportError as exc:
            raise EmbeddingProviderError(
                "httpx is required for OllamaProvider. Install with: pip install httpx",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="dependency_missing",
            ) from exc

        payload = {"model": self._model, "input": texts}
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = httpx.post(
                    self._embed_url,
                    json=payload,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "OllamaProvider timeout (attempt %d/%d): url=%s model=%s timeout=%.1fs",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    self._embed_url,
                    self._model,
                    self._timeout,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise EmbeddingProviderError(
                    (
                        f"Ollama request timed out after {self._timeout}s "
                        f"(url={self._embed_url}, model={self._model})"
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="timeout",
                ) from exc
            except httpx.ConnectError as exc:
                last_exc = exc
                logger.warning(
                    "OllamaProvider connection error (attempt %d/%d): url=%s model=%s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    self._embed_url,
                    self._model,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise EmbeddingProviderError(
                    (
                        f"Cannot connect to Ollama at {self._embed_url} "
                        f"(model={self._model}). "
                        "Check EMBEDDING_BASE_URL and that Ollama is running."
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="connection_error",
                ) from exc
            except httpx.RequestError as exc:
                raise EmbeddingProviderError(
                    f"Ollama HTTP request error: {exc} (url={self._embed_url}, model={self._model})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="request_error",
                ) from exc

            # HTTP response received
            if response.status_code in _RETRYABLE_STATUS:
                last_exc = None
                logger.warning(
                    "OllamaProvider HTTP %d (attempt %d/%d): url=%s model=%s",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    self._embed_url,
                    self._model,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise EmbeddingProviderError(
                    (
                        f"Ollama returned HTTP {response.status_code} after "
                        f"{_MAX_RETRIES + 1} attempts "
                        f"(url={self._embed_url}, model={self._model})"
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="http_error",
                )

            if not response.is_success:
                raise EmbeddingProviderError(
                    (
                        f"Ollama returned HTTP {response.status_code} "
                        f"(url={self._embed_url}, model={self._model})"
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="http_error",
                )

            try:
                data = response.json()
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"Ollama response is not valid JSON (url={self._embed_url}, model={self._model})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="invalid_response",
                ) from exc

            # Validate response shape – these errors are NOT retried
            vectors = _validate_response(
                data,
                expected_count=len(texts),
                provider=_PROVIDER_NAME,
                model=self._model,
                url=self._embed_url,
            )

            # Update discovered dimensions
            if vectors:
                self._discovered_dimensions = len(vectors[0])
                if self._expected_dimensions > 0 and self._discovered_dimensions != self._expected_dimensions:
                    raise EmbeddingDimensionMismatchError(
                        (
                            f"Ollama returned {self._discovered_dimensions}-dim vectors but "
                            f"EMBEDDING_DIMENSIONS={self._expected_dimensions} "
                            f"(url={self._embed_url}, model={self._model})"
                        ),
                        provider=_PROVIDER_NAME,
                        model=self._model,
                        expected=self._expected_dimensions,
                        actual=self._discovered_dimensions,
                    )

            return vectors

        # Should not reach here, but satisfy type checker
        raise EmbeddingProviderError(
            f"Ollama embed failed after retries (url={self._embed_url}, model={self._model})",
            provider=_PROVIDER_NAME,
            model=self._model,
            category="exhausted_retries",
        )
