"""Ollama embedding provider using POST /api/embed.

WARNING: This file re-exports ``OllamaProvider`` from rag-core.
Do NOT edit the implementation here — modify ``rag_core.embeddings.ollama_provider`` instead.
"""
from __future__ import annotations

from rag_core.embeddings.ollama_provider import (  # noqa: F401
    OllamaProvider,
    _is_finite,
    _validate_response,
)

__all__ = [
    "OllamaProvider",
    "_is_finite",
    "_validate_response",
]
