"""Celery application entrypoint for Dominic background workers.

This module intentionally imports only configuration and Celery itself. It must
remain importable without importing ``app.main`` or any ``app.api.*`` module so
that worker startup cannot create a FastAPI circular import.
"""
from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "dominic",
    include=["app.worker.tasks.ingestion"],
)

celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_always_eager=settings.celery_task_always_eager,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    result_expires=86400,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_time_limit=settings.celery_task_time_limit,
    task_routes={
        "ingestion.*": {"queue": "ingestion"},
    },
)

# Autodiscover the task package while keeping task modules under app.worker.*.
# The explicit include above ensures app.worker.tasks.ingestion is imported by
# Celery workers even if no other code imports the package first.
celery_app.autodiscover_tasks(["app.worker"], related_name="tasks")

__all__ = ["celery_app"]
