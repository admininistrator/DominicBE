"""Request/response format adapters for API-based embedding providers.

WARNING: This file re-exports all adapters from rag-core.
Do NOT edit the adapters here — modify ``rag_core.embeddings.api_adapters`` instead.
"""
from __future__ import annotations

from rag_core.embeddings.api_adapters import (  # noqa: F401
    APIAdapter,
    CohereAdapter,
    HuggingFaceAdapter,
    NVIDIAAdapter,
    OllamaAPIAdapter,
    OpenAIAdapter,
    VoyageAdapter,
    get_api_adapter,
)

__all__ = [
    "APIAdapter",
    "OpenAIAdapter",
    "NVIDIAAdapter",
    "CohereAdapter",
    "VoyageAdapter",
    "HuggingFaceAdapter",
    "OllamaAPIAdapter",
    "get_api_adapter",
]
