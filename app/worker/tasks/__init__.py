"""Celery task modules for Dominic workers."""

from app.worker.tasks.ingestion import ingest_document_async

__all__ = ["ingest_document_async"]
