"""Custom ingestion pipeline — wraps the existing chunk_text() behavior.

WARNING: This file re-exports ``CustomPipeline`` from rag-core.
Do NOT edit the implementation here — modify ``rag_core.chunking.custom_pipeline`` instead.
"""
from __future__ import annotations

from rag_core.chunking.custom_pipeline import CustomPipeline  # noqa: F401

__all__ = [
    "CustomPipeline",
]
