"""Ingestion pipeline factory — selects pipeline from INGESTION_PIPELINE setting.

Usage:
    from app.services.ingestion.factory import get_ingestion_pipeline
    pipeline = get_ingestion_pipeline()
    chunks = pipeline.chunk_document(normalized_text, document_id=doc.id)

The factory reads settings.ingestion_pipeline and returns the appropriate
IngestionPipeline implementation.  LlamaIndex import is lazy and optional —
if llama-index-core is not installed and INGESTION_PIPELINE=llamaindex, a
clear IngestionPipelineError is raised at call time (not at import time).

No dependency on CRUD, vector_store, endpoints, or chat modules.
"""
from __future__ import annotations

import logging

from app.services.ingestion.base import IngestionPipeline, IngestionPipelineError

logger = logging.getLogger(__name__)

# Supported pipeline identifiers
_PIPELINE_CUSTOM = "custom"
_PIPELINE_LLAMAINDEX = "llamaindex"


def get_ingestion_pipeline(
    *,
    pipeline: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionPipeline:
    """Return the configured ingestion pipeline.

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
    from app.core.config import settings  # local import keeps module importable without settings

    selected = (pipeline or settings.ingestion_pipeline or _PIPELINE_CUSTOM).strip().lower()

    if selected == _PIPELINE_CUSTOM:
        from app.services.ingestion.custom_pipeline import CustomPipeline
        logger.debug("get_ingestion_pipeline: returning CustomPipeline")
        return CustomPipeline()

    if selected == _PIPELINE_LLAMAINDEX:
        # Lazy import — llama-index-core is optional.  If not installed, raise
        # a clear error rather than an ImportError at module load time.
        try:
            from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline
        except ImportError as exc:
            raise IngestionPipelineError(
                f"INGESTION_PIPELINE=llamaindex requires llama-index-core. "
                f"Install with: pip install llama-index-core. "
                f"Original error: {exc}",
                pipeline=selected,
                category="missing_dependency",
            ) from exc
        logger.debug("get_ingestion_pipeline: returning LlamaIndexPipeline")
        return LlamaIndexPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise IngestionPipelineError(
        f"Unknown INGESTION_PIPELINE value: '{selected}'. "
        f"Supported values: '{_PIPELINE_CUSTOM}', '{_PIPELINE_LLAMAINDEX}'.",
        pipeline=selected,
        category="configuration_error",
    )
