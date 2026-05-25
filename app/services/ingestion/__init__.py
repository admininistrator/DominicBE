"""Ingestion pipeline package — public exports only, no side effects.

Usage:
    from app.services.ingestion import IngestionChunk, IngestionPipeline
    from app.services.ingestion import IngestionPipelineError
    from app.services.ingestion.factory import get_ingestion_pipeline
"""
from app.services.ingestion.base import (  # noqa: F401
    IngestionChunk,
    IngestionPipeline,
    IngestionPipelineError,
)

__all__ = [
    "IngestionChunk",
    "IngestionPipeline",
    "IngestionPipelineError",
]
