"""Embedding provider package.

Exports the public surface used by knowledge_service and retrieval_service.
No side effects on import.
"""
from app.services.embeddings.base import (
    EmbeddingDimensionMismatchError,
    EmbeddingMeta,
    EmbeddingProvider,
    EmbeddingProviderCapabilities,
    EmbeddingProviderError,
    EmbedResult,
    QueryEmbedResult,
)
from app.services.embeddings.factory import get_embedding_provider

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
