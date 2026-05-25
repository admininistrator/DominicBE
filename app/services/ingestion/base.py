"""Ingestion pipeline protocol and canonical chunk result shape.

WARNING: This file re-exports from rag-core.
Do NOT edit the protocol here — modify ``rag_core.chunking.base`` instead.
"""
from __future__ import annotations

from rag_core.chunking.base import (  # noqa: F401
    IngestionChunk,
    IngestionPipeline,
    IngestionPipelineError,
)

__all__ = [
    "IngestionChunk",
    "IngestionPipeline",
    "IngestionPipelineError",
]
