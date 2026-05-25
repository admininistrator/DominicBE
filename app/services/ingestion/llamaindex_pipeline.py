"""LlamaIndex ingestion adapter — converts LlamaIndex nodes to canonical IngestionChunk objects.

WARNING: This file re-exports ``LlamaIndexPipeline`` from rag-core.
Do NOT edit the implementation here — modify ``rag_core.chunking.llamaindex_pipeline`` instead.
"""
from __future__ import annotations

from rag_core.chunking.llamaindex_pipeline import LlamaIndexPipeline  # noqa: F401

__all__ = [
    "LlamaIndexPipeline",
]
