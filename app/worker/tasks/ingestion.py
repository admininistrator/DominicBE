"""Celery task skeleton for asynchronous knowledge document ingestion."""
from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _bounded_retry_countdown_seconds(current_retries: int) -> float:
    """Return exponential retry delay bounded by the configured retry count."""
    base_delay = max(float(settings.ingestion_retry_delay_seconds), 0.0)
    max_retries = max(int(settings.ingestion_max_retries), 0)
    bounded_retry_index = min(max(int(current_retries), 0), max_retries)
    max_countdown = base_delay * (2**max_retries)
    return min(base_delay * (2**bounded_retry_index), max_countdown)


def _mark_ingestion_job(job_id: int, status: str, error_message: str | None = None) -> None:
    """Persist ingestion job status using the existing CRUD helper."""
    from app.core.database import SessionLocal
    from app.crud import crud_knowledge

    db = SessionLocal()
    try:
        crud_knowledge.update_ingestion_job_status(
            db,
            job_id,
            status,
            error_message,
        )
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="ingestion.ingest_document_async",
    queue="ingestion",
    max_retries=settings.ingestion_max_retries,
    default_retry_delay=settings.ingestion_retry_delay_seconds,
    soft_time_limit=settings.celery_task_soft_time_limit,
    time_limit=settings.celery_task_time_limit,
)
def ingest_document_async(self: Any, document_id: int, job_id: int) -> dict[str, Any]:
    """Run the existing durable indexing pipeline from a Celery worker.

    Arguments are intentionally primitive JSON-serializable values. The task
    opens fresh database sessions inside the worker process and delegates all
    indexing behavior to ``run_indexing_pipeline`` so rag-core remains unaware
    of Celery.
    """
    document_id = int(document_id)
    job_id = int(job_id)

    from app.core.database import SessionLocal

    try:
        _mark_ingestion_job(job_id, "processing")

        # Import service code lazily to avoid Celery/FastAPI circular imports.
        from app.services.knowledge_service import run_indexing_pipeline

        return run_indexing_pipeline(document_id, job_id, SessionLocal)
    except Exception as exc:
        error_message = str(exc)
        try:
            _mark_ingestion_job(job_id, "failed", error_message)
        except Exception:
            logger.exception(
                "Failed to persist failed status for ingestion job=%s document=%s",
                job_id,
                document_id,
            )

        current_retries = int(getattr(self.request, "retries", 0) or 0)
        max_retries = max(int(settings.ingestion_max_retries), 0)
        if current_retries >= max_retries:
            logger.exception(
                "Ingestion task exhausted retries for document=%s job=%s",
                document_id,
                job_id,
            )
            raise

        countdown = _bounded_retry_countdown_seconds(current_retries)
        logger.warning(
            "Retrying ingestion task for document=%s job=%s in %.1f seconds (%s/%s retries used): %s",
            document_id,
            job_id,
            countdown,
            current_retries,
            max_retries,
            error_message,
        )
        raise self.retry(
            exc=exc,
            countdown=countdown,
            max_retries=max_retries,
        )
