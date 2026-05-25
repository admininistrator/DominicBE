"""Embedding provider protocol and shared result types.

WARNING: This file now re-exports from rag-core. Do NOT edit the protocol
types here — modify ``rag_core.embeddings.base`` instead.

Provides backward-compatible imports for all existing DominicBE code.
"""
from __future__ import annotations

from rag_core.embeddings.base import (  # noqa: F401
    EmbeddingDimensionMismatchError,
    EmbeddingMeta,
    EmbeddingProvider,
    EmbeddingProviderCapabilities,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)

__all__ = [
    "EmbeddingDimensionMismatchError",
    "EmbeddingMeta",
    "EmbeddingProvider",
    "EmbeddingProviderCapabilities",
    "EmbeddingProviderError",
    "EmbedResult",
    "QueryEmbedResult",
]
