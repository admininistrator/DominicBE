"""Ingestion pipeline factory — selects pipeline from INGESTION_PIPELINE setting.

WARNING: This file now delegates to ``rag_core.chunking.factory``.
Do NOT edit pipeline selection logic here — modify ``rag_core.chunking.factory`` instead.
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.services.ingestion.base import IngestionPipeline, IngestionPipelineError
from rag_core.chunking.factory import get_ingestion_pipeline as _rag_core_get_ingestion_pipeline

logger = logging.getLogger(__name__)


def get_ingestion_pipeline(
    *,
    pipeline: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionPipeline:
    """Return the configured ingestion pipeline.

    Delegates to ``rag_core.chunking.factory.get_ingestion_pipeline`` with
    defaults from DominicBE's ``settings``.

    Args:
        pipeline:      Override for settings.ingestion_pipeline.
                       Defaults to the value from app.core.config.settings.
        chunk_size:    Optional chunk size override (passed to the pipeline).
        chunk_overlap: Optional chunk overlap override (passed to the pipeline).

    Returns:
        An IngestionPipeline implementation.

    Raises:
        IngestionPipelineError: If the pipeline name is unknown or if the
            required dependency (e.g. llama-index-core) is not installed.
    """
    return _rag_core_get_ingestion_pipeline(
        pipeline=(pipeline or settings.ingestion_pipeline or "custom"),
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )
