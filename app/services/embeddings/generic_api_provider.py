"""Generic OpenAI-compatible API embedding provider.

This provider connects to any OpenAI-compatible embedding API (and format
adapters for Cohere, Voyage, HuggingFace, Ollama) entirely via configuration.

It implements the ``EmbeddingProvider`` protocol from
:mod:`app.services.embeddings.base`.

Design:
- Uses ``httpx`` for HTTP calls (already in requirements.txt).
- Delegates request/response format to an ``APIAdapter`` selected by
  ``EMBEDDING_API_TYPE``.
- Retries transient failures (5xx, 429, timeout, connection error) with
  exponential backoff — same policy as :class:`OllamaProvider`.
- Does NOT retry validation failures or 4xx (except 429).
- Logs latency, provider, model, dims, batch_size — never logs raw text or
  API key.

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import json
import logging
import math
import time
from urllib.parse import urljoin

from app.services.embeddings.api_adapters import APIAdapter, get_api_adapter
from app.services.embeddings.base import (
    EmbeddingDimensionMismatchError,
    EmbeddingMeta,
    EmbeddingProviderCapabilities,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)
from app.services.embeddings.security import (
    mask_api_key,
    sanitize_error_message,
    validate_api_key,
)

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "api"
_BASE_VERSION = "api-v1"

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


class GenericAPIProvider:
    """Generic OpenAI-compatible API embedding provider.

    Args:
        model: Model name from config (e.g. ``"text-embedding-3-small"``).
        base_url: API base URL (e.g. ``"https://api.openai.com/v1"``).
        api_key: API key for authentication (sensitive, never logged).
        api_type: API format type (e.g. ``"openai"``, ``"cohere"``,
            ``"voyage"``, ``"huggingface"``, ``"ollama"``, or empty).
        timeout_seconds: HTTP timeout for each request.
        batch_size: Maximum texts per single API call.
        expected_dimensions: If > 0, validate that returned vectors match.
        api_version: Optional API version string (e.g. ``"2024-02-01"``).
        custom_headers: Optional dict of custom HTTP headers.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "",
        api_key: str = "",
        api_type: str = "",
        timeout_seconds: float = 60.0,
        batch_size: int = 16,
        expected_dimensions: int = 0,
        api_version: str = "",
        custom_headers: dict | None = None,
    ) -> None:
        self._model = model
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key
        self._api_type = (api_type or "").strip().lower() or "openai"
        self._timeout = timeout_seconds
        self._batch_size = max(1, batch_size)
        self._expected_dimensions = expected_dimensions
        self._api_version = api_version
        self._custom_headers = custom_headers or {}
        self._adapter: APIAdapter = get_api_adapter(self._api_type)
        self._version = f"{_BASE_VERSION.split('-v1')[0]}-{self._api_type}-v1"
        # Dimensions are discovered on first successful call when not pre-configured
        self._discovered_dimensions: int = expected_dimensions

        # Validate API key on construction
        validate_api_key(self._api_key, _PROVIDER_NAME, self._api_type)

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Embed a batch of texts via the configured API provider.

        Splits into sub-batches of at most ``batch_size`` texts.

        Args:
            texts: Non-empty ordered list of text strings.

        Returns:
            ``EmbedResult`` with vectors in the same order as input texts.

        Raises:
            EmbeddingProviderError: On any provider-level failure.
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

        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            batch_vectors = self._call_api(batch)
            all_vectors.extend(batch_vectors)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        dims = len(all_vectors[0]) if all_vectors else self._discovered_dimensions

        logger.info(
            "GenericAPIProvider.embed_texts: provider=%s model=%s api_type=%s dims=%d batch=%d latency_ms=%.1f",
            _PROVIDER_NAME,
            self._model,
            self._api_type,
            dims,
            len(texts),
            latency_ms,
        )

        meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=dims,
            version=self._version,
            extra={
                "latency_ms": latency_ms,
                "batch_size": len(texts),
                "api_type": self._api_type,
            },
        )
        return EmbedResult(vectors=all_vectors, meta=meta)

    def embed_query(self, query: str) -> QueryEmbedResult:
        """Embed a single query string via the configured API provider.

        Args:
            query: Non-empty query text.

        Returns:
            ``QueryEmbedResult`` with a single vector.

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
            "GenericAPIProvider.embed_query: provider=%s model=%s api_type=%s dims=%d latency_ms=%.1f",
            _PROVIDER_NAME,
            self._model,
            self._api_type,
            dims,
            latency_ms,
        )

        meta = EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=dims,
            version=self._version,
            extra={
                "latency_ms": latency_ms,
                "api_type": self._api_type,
            },
        )
        return QueryEmbedResult(vector=vector, meta=meta)

    @property
    def meta(self) -> EmbeddingMeta:
        return EmbeddingMeta(
            provider=_PROVIDER_NAME,
            model=self._model,
            dimensions=self._discovered_dimensions,
            version=self._version,
        )

    @property
    def capabilities(self) -> EmbeddingProviderCapabilities:
        """Report capabilities based on the active adapter."""
        return EmbeddingProviderCapabilities(
            supports_batch=True,
            max_batch_size=self._batch_size,
            supports_truncation=True,
            supports_dimensions_param=(self._api_type in ("openai", "cohere", "voyage")),
            requires_api_key=(self._api_type in ("openai", "cohere", "voyage")),
            api_type=self._api_type,
        )

    # ------------------------------------------------------------------
    # Internal HTTP call with retry
    # ------------------------------------------------------------------

    def _build_url(self, endpoint_path: str) -> str:
        """Build the full URL from base URL and endpoint path.

        Args:
            endpoint_path: Path returned by the adapter (e.g. ``"/v1/embeddings"``).

        Returns:
            Full URL string.
        """
        if endpoint_path:
            return urljoin(f"{self._base_url}/", endpoint_path.lstrip("/"))
        return self._base_url

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the API request.

        Returns:
            Dict of HTTP headers.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._api_version:
            headers["Api-Version"] = self._api_version
        # Merge custom headers (allow override of defaults)
        for key, value in self._custom_headers.items():
            headers[key] = str(value)
        return headers

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Execute an API call with retry on transient failures.

        Args:
            texts: Batch of texts (already bounded by ``batch_size``).

        Returns:
            List of float vectors in input order.

        Raises:
            EmbeddingProviderError: On non-retryable failure or exhausted retries.
        """
        try:
            import httpx
        except ImportError as exc:
            raise EmbeddingProviderError(
                "httpx is required for GenericAPIProvider.",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="dependency_missing",
            ) from exc

        # Build request via adapter
        endpoint_path, request_body = self._adapter.format_request(
            texts, self._model, self._discovered_dimensions or None
        )
        url = self._build_url(endpoint_path)
        headers = self._build_headers()

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = httpx.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "GenericAPIProvider timeout (attempt %d/%d): host=%s model=%s api_type=%s timeout=%.1fs",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    self._get_host(url),
                    self._model,
                    self._api_type,
                    self._timeout,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise EmbeddingProviderError(
                    (
                        f"API request timed out after {self._timeout}s "
                        f"(host={self._get_host(url)}, model={self._model}, "
                        f"api_type={self._api_type})"
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="timeout",
                ) from exc
            except httpx.ConnectError as exc:
                last_exc = exc
                logger.warning(
                    "GenericAPIProvider connection error (attempt %d/%d): host=%s model=%s api_type=%s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    self._get_host(url),
                    self._model,
                    self._api_type,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise EmbeddingProviderError(
                    (
                        f"Cannot connect to API at {self._get_host(url)} "
                        f"(model={self._model}, api_type={self._api_type}). "
                        "Check EMBEDDING_BASE_URL."
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="connection_error",
                ) from exc
            except httpx.RequestError as exc:
                raise EmbeddingProviderError(
                    f"API HTTP request error: {exc} "
                    f"(host={self._get_host(url)}, model={self._model}, "
                    f"api_type={self._api_type})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="request_error",
                ) from exc

            # HTTP response received
            if response.status_code in _RETRYABLE_STATUS:
                last_exc = None
                logger.warning(
                    "GenericAPIProvider HTTP %d (attempt %d/%d): host=%s model=%s api_type=%s",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    self._get_host(url),
                    self._model,
                    self._api_type,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise EmbeddingProviderError(
                    (
                        f"API returned HTTP {response.status_code} after "
                        f"{_MAX_RETRIES + 1} attempts "
                        f"(host={self._get_host(url)}, model={self._model}, "
                        f"api_type={self._api_type})"
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="http_error",
                )

            if not response.is_success:
                # Sanitize error message before raising
                error_body = ""
                try:
                    error_body = response.text[:500]
                except Exception:
                    pass
                sanitized = sanitize_error_message(error_body, self._api_key)
                raise EmbeddingProviderError(
                    (
                        f"API returned HTTP {response.status_code} "
                        f"(host={self._get_host(url)}, model={self._model}, "
                        f"api_type={self._api_type})"
                        + (f" body={sanitized}" if sanitized else "")
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="http_error",
                )

            try:
                data = response.json()
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"API response is not valid JSON "
                    f"(host={self._get_host(url)}, model={self._model}, "
                    f"api_type={self._api_type})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="invalid_response",
                ) from exc

            # Parse response through adapter — these errors are NOT retried
            try:
                vectors = self._adapter.parse_response(data, len(texts))
            except ValueError as exc:
                raise EmbeddingProviderError(
                    f"Failed to parse API response: {exc} "
                    f"(host={self._get_host(url)}, model={self._model}, "
                    f"api_type={self._api_type})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="invalid_response",
                ) from exc

            # Validate parsed vectors
            vectors = self._validate_vectors(vectors, url)

            # Update discovered dimensions
            if vectors:
                self._discovered_dimensions = len(vectors[0])
                if self._expected_dimensions > 0 and self._discovered_dimensions != self._expected_dimensions:
                    raise EmbeddingDimensionMismatchError(
                        (
                            f"API returned {self._discovered_dimensions}-dim vectors but "
                            f"EMBEDDING_DIMENSIONS={self._expected_dimensions} "
                            f"(host={self._get_host(url)}, model={self._model}, "
                            f"api_type={self._api_type})"
                        ),
                        provider=_PROVIDER_NAME,
                        model=self._model,
                        expected=self._expected_dimensions,
                        actual=self._discovered_dimensions,
                    )

            return vectors

        # Should not reach here, but satisfy type checker
        raise EmbeddingProviderError(
            f"API embed failed after retries "
            f"(host={self._get_host(url)}, model={self._model})",
            provider=_PROVIDER_NAME,
            model=self._model,
            category="exhausted_retries",
        )

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        url: str,
    ) -> list[list[float]]:
        """Validate parsed embedding vectors.

        Checks:
        - All vectors non-empty.
        - All values are finite numbers.
        - All vectors have consistent dimensions.

        Args:
            vectors: Parsed embedding vectors.
            url: The request URL (for error messages).

        Returns:
            The same vectors if valid.

        Raises:
            EmbeddingProviderError: On any validation failure.
            EmbeddingDimensionMismatchError: On inconsistent dimensions.
        """
        if not vectors:
            raise EmbeddingProviderError(
                f"API returned zero embeddings "
                f"(host={self._get_host(url)}, model={self._model}, "
                f"api_type={self._api_type})",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="invalid_response",
            )

        first_dim = len(vectors[0])
        if first_dim == 0:
            raise EmbeddingProviderError(
                f"API returned empty embedding vector at index 0 "
                f"(host={self._get_host(url)}, model={self._model}, "
                f"api_type={self._api_type})",
                provider=_PROVIDER_NAME,
                model=self._model,
                category="invalid_response",
            )

        for idx, vec in enumerate(vectors):
            if not vec:
                raise EmbeddingProviderError(
                    f"API returned empty embedding vector at index {idx} "
                    f"(host={self._get_host(url)}, model={self._model}, "
                    f"api_type={self._api_type})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="invalid_response",
                )
            if not all(_is_finite(v) for v in vec):
                raise EmbeddingProviderError(
                    f"API embedding[{idx}] contains non-finite values "
                    f"(host={self._get_host(url)}, model={self._model}, "
                    f"api_type={self._api_type})",
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    category="non_numeric_values",
                )
            dim = len(vec)
            if dim != first_dim:
                raise EmbeddingDimensionMismatchError(
                    (
                        f"API embedding[{idx}] has dimension {dim} but embedding[0] "
                        f"has dimension {first_dim} "
                        f"(host={self._get_host(url)}, model={self._model}, "
                        f"api_type={self._api_type})"
                    ),
                    provider=_PROVIDER_NAME,
                    model=self._model,
                    expected=first_dim,
                    actual=dim,
                )

        return vectors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_host(url: str) -> str:
        """Extract host from URL for logging (no path, no API key)."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            host = parsed.hostname or url
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return host
        except Exception:
            return url.split("/")[0] if "/" in url else url
