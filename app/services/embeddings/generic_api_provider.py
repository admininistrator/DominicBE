"""Generic OpenAI-compatible API embedding provider.

WARNING: This file re-exports ``GenericAPIProvider`` from rag-core.
Do NOT edit the implementation here — modify ``rag_core.embeddings.generic_api_provider`` instead.
"""
from __future__ import annotations

from rag_core.embeddings.generic_api_provider import (  # noqa: F401
    GenericAPIProvider,
    _is_finite,
)

__all__ = [
    "GenericAPIProvider",
    "_is_finite",
]
