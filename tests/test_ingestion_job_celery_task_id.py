from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
from datetime import datetime
from types import SimpleNamespace

from fastapi import BackgroundTasks


os.environ["DEBUG"] = "false"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.crud import crud_knowledge
from app.models.chat_models import ChatSession  # noqa: F401 - register FK target table
from app.models.knowledge_models import IngestionJob, KnowledgeDocument
from app.schemas.knowledge_schemas import IngestionJobResponse, KnowledgeDocumentCreateRequest


class _FakeUpload:
    filename = "async-celery.txt"
    content_type = "text/plain"

    def __init__(self, content: bytes = b"hello celery upload"):
        self._content = content

    async def read(self) -> bytes:
        return self._content


def _import_knowledge_endpoint(monkeypatch):
    """Import knowledge endpoint despite the local FastAPI/Starlette mismatch.

    The project venv currently has a known pre-existing fastapi/starlette
    incompatibility where FastAPI passes on_startup/on_shutdown to Starlette's
    Router. The endpoint logic under test is independent of that constructor
    argument plumbing, so this shim keeps the regression focused on Phase D.
    """
    from starlette.routing import Router

    if "on_startup" not in inspect.signature(Router.__init__).parameters:
        original_init = Router.__init__

        def compat_init(self, *args, **kwargs):
            kwargs.pop("on_startup", None)
            kwargs.pop("on_shutdown", None)
            return original_init(self, *args, **kwargs)

        monkeypatch.setattr(Router, "__init__", compat_init)

    from app.api.endpoints import knowledge as knowledge_endpoint

    return knowledge_endpoint


def _session_factory():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, db_path, SessionLocal


def test_ingestion_job_celery_task_id_defaults_to_none_and_can_be_updated():
    assert hasattr(IngestionJob, "celery_task_id")

    engine, db_path, SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        doc = KnowledgeDocument(
            owner_username="tester",
            title="Async Test",
            source_type="text",
            raw_text="hello",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        job = crud_knowledge.create_ingestion_job(db, doc.id)
        assert job.celery_task_id is None

        updated = crud_knowledge.set_ingestion_job_celery_task_id(
            db,
            job.id,
            "celery-task-123",
        )

        assert updated is not None
        assert updated.celery_task_id == "celery-task-123"
        assert crud_knowledge.get_ingestion_job(db, job.id).celery_task_id == "celery-task-123"
    finally:
        db.close()
        engine.dispose()
        os.remove(db_path)


def test_ingestion_job_response_exposes_celery_task_id():
    response = IngestionJobResponse.model_validate(
        IngestionJob(
            id=1,
            document_id=2,
            status="queued",
            celery_task_id="celery-task-abc",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
    )

    assert response.celery_task_id == "celery-task-abc"
    assert response.model_dump()["celery_task_id"] == "celery-task-abc"


def test_text_ingest_celery_async_branch_dispatches_and_returns_queued(monkeypatch):
    knowledge_endpoint = _import_knowledge_endpoint(monkeypatch)

    engine, db_path, SessionLocal = _session_factory()
    db = SessionLocal()
    dispatched: list[tuple[int, int]] = []

    def fake_dispatch(document_id: int, job_id: int) -> str:
        dispatched.append((document_id, job_id))
        return "celery-task-text-123"

    monkeypatch.setattr(knowledge_endpoint.settings, "celery_enabled", True)
    import app.services.knowledge_service as knowledge_service
    monkeypatch.setattr(knowledge_service.object_storage, "is_object_storage_enabled", lambda: False)
    monkeypatch.setattr(
        knowledge_endpoint,
        "_dispatch_ingestion_job_to_celery",
        fake_dispatch,
        raising=False,
    )

    background_tasks = BackgroundTasks()
    try:
        result = knowledge_endpoint.ingest_text(
            request=KnowledgeDocumentCreateRequest(
                title="Async Text",
                raw_text="hello celery text",
                source_type="text",
            ),
            background_tasks=background_tasks,
            async_index=True,
            current_user=SimpleNamespace(username="tester"),
            db=db,
        )

        assert result.status == "queued"
        assert result.celery_task_id == "celery-task-text-123"
        assert dispatched == [(result.document_id, result.job_id)]
        assert len(background_tasks.tasks) == 0
        assert (
            crud_knowledge.get_ingestion_job(db, result.job_id).celery_task_id
            == "celery-task-text-123"
        )
    finally:
        db.close()
        engine.dispose()
        os.remove(db_path)


def test_upload_celery_async_branch_dispatches_and_returns_queued(monkeypatch):
    knowledge_endpoint = _import_knowledge_endpoint(monkeypatch)

    engine, db_path, SessionLocal = _session_factory()
    db = SessionLocal()
    dispatched: list[tuple[int, int]] = []

    def fake_dispatch(document_id: int, job_id: int) -> str:
        dispatched.append((document_id, job_id))
        return "celery-task-upload-456"

    monkeypatch.setattr(knowledge_endpoint.settings, "celery_enabled", True)
    import app.services.knowledge_service as knowledge_service
    monkeypatch.setattr(knowledge_service.object_storage, "is_object_storage_enabled", lambda: False)
    monkeypatch.setattr(
        knowledge_endpoint,
        "_dispatch_ingestion_job_to_celery",
        fake_dispatch,
        raising=False,
    )

    background_tasks = BackgroundTasks()
    try:
        result = asyncio.run(
            knowledge_endpoint.upload_document(
                background_tasks=background_tasks,
                file=_FakeUpload(),
                async_index=True,
                current_user=SimpleNamespace(username="tester"),
                db=db,
            )
        )

        assert result.status == "queued"
        assert result.celery_task_id == "celery-task-upload-456"
        assert dispatched == [(result.document_id, result.job_id)]
        assert len(background_tasks.tasks) == 0
        assert (
            crud_knowledge.get_ingestion_job(db, result.job_id).celery_task_id
            == "celery-task-upload-456"
        )
    finally:
        db.close()
        engine.dispose()
        os.remove(db_path)
