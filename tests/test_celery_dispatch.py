from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fastapi import BackgroundTasks


os.environ["DEBUG"] = "false"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.crud import crud_knowledge
from app.models.chat_models import ChatSession  # noqa: F401 - register FK target table
from app.schemas.knowledge_schemas import KnowledgeDocumentCreateRequest


class _FakeUpload:
    filename = "async-celery.txt"
    content_type = "text/plain"

    def __init__(self, content: bytes = b"hello celery upload"):
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _FakeAsyncResult:
    id = "celery-task-from-delay"


def _import_knowledge_endpoint(monkeypatch):
    """Import knowledge endpoint despite the local FastAPI/Starlette mismatch."""
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


def _disable_object_storage(monkeypatch):
    import app.services.knowledge_service as knowledge_service

    monkeypatch.setattr(knowledge_service.object_storage, "is_object_storage_enabled", lambda: False)


def _patch_celery_delay(monkeypatch):
    from app.worker.tasks import ingestion as ingestion_task

    calls: list[tuple[int, int]] = []

    def fake_delay(document_id: int, job_id: int):
        calls.append((document_id, job_id))
        return _FakeAsyncResult()

    monkeypatch.setattr(ingestion_task.ingest_document_async, "delay", fake_delay)
    return calls


def test_upload_async_with_celery_enabled_dispatches_delay_and_persists_task_id(monkeypatch):
    knowledge_endpoint = _import_knowledge_endpoint(monkeypatch)
    _disable_object_storage(monkeypatch)
    delay_calls = _patch_celery_delay(monkeypatch)
    monkeypatch.setattr(knowledge_endpoint.settings, "celery_enabled", True)

    engine, db_path, SessionLocal = _session_factory()
    db = SessionLocal()
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
        assert result.celery_task_id == "celery-task-from-delay"
        assert delay_calls == [(result.document_id, result.job_id)]
        assert len(background_tasks.tasks) == 0
        assert (
            crud_knowledge.get_ingestion_job(db, result.job_id).celery_task_id
            == "celery-task-from-delay"
        )
    finally:
        db.close()
        engine.dispose()
        os.remove(db_path)


def test_upload_async_with_celery_disabled_uses_background_tasks_without_dispatch(monkeypatch):
    knowledge_endpoint = _import_knowledge_endpoint(monkeypatch)
    _disable_object_storage(monkeypatch)
    delay_calls = _patch_celery_delay(monkeypatch)
    monkeypatch.setattr(knowledge_endpoint.settings, "celery_enabled", False)

    engine, db_path, SessionLocal = _session_factory()
    db = SessionLocal()
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

        assert result.status == "pending"
        assert result.celery_task_id is None
        assert delay_calls == []
        assert len(background_tasks.tasks) == 1
        assert crud_knowledge.get_ingestion_job(db, result.job_id).celery_task_id is None
    finally:
        db.close()
        engine.dispose()
        os.remove(db_path)


def test_text_async_with_celery_enabled_dispatches_delay_and_persists_task_id(monkeypatch):
    knowledge_endpoint = _import_knowledge_endpoint(monkeypatch)
    _disable_object_storage(monkeypatch)
    delay_calls = _patch_celery_delay(monkeypatch)
    monkeypatch.setattr(knowledge_endpoint.settings, "celery_enabled", True)

    engine, db_path, SessionLocal = _session_factory()
    db = SessionLocal()
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
        assert result.celery_task_id == "celery-task-from-delay"
        assert delay_calls == [(result.document_id, result.job_id)]
        assert len(background_tasks.tasks) == 0
        assert (
            crud_knowledge.get_ingestion_job(db, result.job_id).celery_task_id
            == "celery-task-from-delay"
        )
    finally:
        db.close()
        engine.dispose()
        os.remove(db_path)


def test_text_sync_when_celery_enabled_and_async_index_false_preserves_sync_path(monkeypatch):
    knowledge_endpoint = _import_knowledge_endpoint(monkeypatch)
    delay_calls = _patch_celery_delay(monkeypatch)
    monkeypatch.setattr(knowledge_endpoint.settings, "celery_enabled", True)
    sync_calls: list[dict] = []

    def fake_ingest_document(**kwargs):
        sync_calls.append(kwargs)
        return {"document_id": 11, "job_id": 22, "status": "indexed", "chunks_count": 1}

    monkeypatch.setattr(knowledge_endpoint, "ingest_document", fake_ingest_document)
    background_tasks = BackgroundTasks()

    result = knowledge_endpoint.ingest_text(
        request=KnowledgeDocumentCreateRequest(
            title="Sync Text",
            raw_text="hello sync text",
            source_type="text",
        ),
        background_tasks=background_tasks,
        async_index=False,
        current_user=SimpleNamespace(username="tester"),
        db=object(),
    )

    assert result.status == "indexed"
    assert result.celery_task_id is None
    assert delay_calls == []
    assert len(background_tasks.tasks) == 0
    assert sync_calls[0]["title"] == "Sync Text"
    assert sync_calls[0]["raw_text"] == "hello sync text"


def test_smoke_redis_reads_broker_url_from_env_and_pings(monkeypatch, capsys):
    smoke_redis = importlib.import_module("scripts.smoke_redis")
    seen_urls: list[str] = []

    class FakeRedisClient:
        def ping(self):
            return True

    def fake_from_url(url: str, socket_connect_timeout: float, socket_timeout: float):
        seen_urls.append(url)
        assert socket_connect_timeout > 0
        assert socket_timeout > 0
        return FakeRedisClient()

    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(smoke_redis.redis.Redis, "from_url", fake_from_url)

    assert smoke_redis.main([]) == 0
    assert seen_urls == ["redis://localhost:6379/0"]
    assert "PASS" in capsys.readouterr().out


def test_smoke_redis_fails_without_broker_url(monkeypatch, capsys):
    smoke_redis = importlib.import_module("scripts.smoke_redis")
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)

    assert smoke_redis.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_smoke_celery_worker_passes_when_any_worker_responds(monkeypatch, capsys):
    smoke_celery_worker = importlib.import_module("scripts.smoke_celery_worker")

    class FakeInspector:
        def ping(self):
            return {"worker1": {"ok": "pong"}}

    class FakeControl:
        def inspect(self, timeout: float):
            assert timeout > 0
            return FakeInspector()

    monkeypatch.setattr(smoke_celery_worker.celery_app, "control", FakeControl())

    assert smoke_celery_worker.main([]) == 0
    assert "PASS" in capsys.readouterr().out


def test_smoke_scripts_help_when_executed_by_path():
    backend_root = Path(__file__).resolve().parents[1]
    for script_name in ("smoke_redis.py", "smoke_celery_worker.py"):
        result = subprocess.run(
            [sys.executable, str(backend_root / "scripts" / script_name), "--help"],
            cwd=backend_root,
            capture_output=True,
            text=True,
            timeout=20,
        )

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_smoke_celery_worker_fails_when_no_worker_responds(monkeypatch, capsys):
    smoke_celery_worker = importlib.import_module("scripts.smoke_celery_worker")

    class FakeInspector:
        def ping(self):
            return {}

    class FakeControl:
        def inspect(self, timeout: float):
            return FakeInspector()

    monkeypatch.setattr(smoke_celery_worker.celery_app, "control", FakeControl())

    assert smoke_celery_worker.main([]) == 1
    assert "FAIL" in capsys.readouterr().out
