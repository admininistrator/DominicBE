"""Local deterministic hash embedding provider (local-hash-v1).

WARNING: This file re-exports ``LocalHashProvider`` from rag-core.
Do NOT edit the implementation here — modify ``rag_core.embeddings.local_hash_provider`` instead.

Backward-compatibility guarantee:
- Same deterministic output for the same input text.
- Same default dimension (64, from EMBEDDING_DIMENSIONS).
- Provider name: 'local', version: 'local-hash-v1'.
"""
from __future__ import annotations

from rag_core.embeddings.local_hash_provider import (  # noqa: F401
    LocalHashProvider,
    _hash_embed,
    _normalize_text,
)

__all__ = [
    "LocalHashProvider",
    "_hash_embed",
    "_normalize_text",
]
