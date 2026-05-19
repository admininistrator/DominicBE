"""Embedding provider protocol and shared result types.

This module defines the contract that all embedding providers must satisfy.
It has no dependency on CRUD, endpoints, vector store, chat, or LlamaIndex modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingMeta:
    """Provenance metadata attached to every embedding result."""

    provider: str
    """Short provider identifier, e.g. 'local' or 'ollama'."""

    model: str
    """Model name used to produce the embedding, e.g. 'local-hash-v1'."""

    dimensions: int
    """Length of each embedding vector."""

    version: str
    """Implementation version string, e.g. 'local-hash-v1' or 'ollama-qwen3-embedding-06b-v1'."""

    extra: dict = field(default_factory=dict)
    """Optional extra metadata (latency_ms, batch_size, etc.)."""


@dataclass
class EmbedResult:
    """Result of a batch embed_texts() call."""

    vectors: list[list[float]]
    """Ordered list of embedding vectors; len(vectors) == len(input texts)."""

    meta: EmbeddingMeta
    """Provenance metadata for this batch."""


@dataclass
class QueryEmbedResult:
    """Result of a single embed_query() call."""

    vector: list[float]
    """Single embedding vector for the query."""

    meta: EmbeddingMeta
    """Provenance metadata."""


# ---------------------------------------------------------------------------
# Capabilities descriptor (optional, for introspection)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingProviderCapabilities:
    """Describes the capabilities of an embedding provider.

    Intended for use by the factory, health checks, and diagnostics to
    introspect provider features without instantiating the provider.

    Every field has a safe default so that providers that do not implement
    the ``capabilities`` property fall back gracefully.
    """

    supports_batch: bool = True
    """Whether the provider supports batch embedding of multiple texts in one call."""

    max_batch_size: int = 256
    """Maximum number of texts per single batch call (0 = no limit)."""

    supports_truncation: bool = False
    """Whether the provider supports automatic truncation of long inputs."""

    supports_dimensions_param: bool = False
    """Whether the provider accepts a ``dimensions`` parameter in the request body."""

    requires_api_key: bool = False
    """Whether the provider strictly requires an API key for operation."""

    api_type: str = ""
    """API format type string (e.g. ``"openai"``, ``"cohere"``)."""


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all embedding providers must implement.

    Providers must not import CRUD, endpoints, vector_store, chat, or LlamaIndex.
    """

    def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Embed a batch of document texts.

        Args:
            texts: Non-empty ordered list of text strings to embed.

        Returns:
            EmbedResult with vectors in the same order as input texts.

        Raises:
            EmbeddingProviderError: On any provider-level failure.
        """
        ...

    def embed_query(self, query: str) -> QueryEmbedResult:
        """Embed a single query string.

        Args:
            query: Non-empty query text.

        Returns:
            QueryEmbedResult with a single vector.

        Raises:
            EmbeddingProviderError: On any provider-level failure.
        """
        ...

    @property
    def meta(self) -> EmbeddingMeta:
        """Return provider metadata (provider, model, dimensions, version)."""
        ...

    @property
    def capabilities(self) -> Optional[EmbeddingProviderCapabilities]:
        """Return provider capabilities descriptor, or None if not implemented.

        This is an OPTIONAL property. Providers that do not implement it
        will return None, which the factory and health checks treat as
        "unknown capabilities — use safe defaults."
        """
        return None


# ---------------------------------------------------------------------------
# Provider exception
# ---------------------------------------------------------------------------

class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider encounters a non-retryable failure.

    Attributes:
        provider: Short provider name.
        model: Model name.
        category: Failure category string (e.g. 'timeout', 'invalid_response').
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.category = category


class EmbeddingDimensionMismatchError(EmbeddingProviderError):
    """Raised when the provider returns vectors with unexpected dimensions."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        expected: int = 0,
        actual: int = 0,
    ) -> None:
        super().__init__(message, provider=provider, model=model, category="dimension_mismatch")
        self.expected = expected
        self.actual = actual
