"""Embedding provider package.

Exports the public surface used by knowledge_service and retrieval_service.
All types are re-exported from ``rag_core.embeddings`` for backward compatibility.
No side effects on import.
"""
from rag_core.embeddings import (  # noqa: F401
    EmbeddingDimensionMismatchError,
    EmbeddingMeta,
    EmbeddingProvider,
    EmbeddingProviderCapabilities,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)
from app.services.embeddings.factory import get_embedding_provider  # noqa: F401

__all__ = [
    "EmbeddingDimensionMismatchError",
    "EmbeddingMeta",
    "EmbeddingProvider",
    "EmbeddingProviderCapabilities",
    "EmbeddingProviderError",
    "EmbedResult",
    "QueryEmbedResult",
    "get_embedding_provider",
]
