from __future__ import annotations

import os
import tempfile

os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.knowledge_models import KnowledgeChunk, KnowledgeDocument
from scripts.backfill_section_metadata import run_backfill


def _factory():
    import app.models.chat_models  # noqa: F401
    import app.models.system_models  # noqa: F401

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return session_factory, engine, db_path


def _seed_document(db, *, owner="alice", title="Legacy Practice", metadata=None, raw_text="Bài thực hành số 4\n1. Nội dung bài một."):
    doc = KnowledgeDocument(
        owner_username=owner,
        title=title,
        source_type="text",
        raw_text=raw_text,
        status="indexed",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    db.add(
        KnowledgeChunk(
            document_id=doc.id,
            chunk_index=0,
            content="Legacy chunk",
            token_count=2,
            metadata_json=metadata or {},
        )
    )
    db.commit()
    return int(doc.id)


def test_section_metadata_backfill_dry_run_identifies_candidate_without_modifying_chunks():
    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            doc_id = _seed_document(db, metadata={"embedding_provider": "local"})

            summary = run_backfill(db, document_ids=[doc_id], apply=False)

            assert summary["mode"] == "dry_run"
            assert summary["selected_documents"] == 1
            assert summary["needs_reindex_count"] == 1
            assert summary["reindexed_count"] == 0
            result = summary["results"][0]
            assert result["status"] == "dry_run"
            assert result["reason"] == "missing_section_and_span_metadata"
            chunk = db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id).one()
            assert chunk.metadata_json == {"embedding_provider": "local"}
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_section_metadata_backfill_dry_run_respects_owner_and_skips_up_to_date_docs():
    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            _seed_document(db, owner="alice", title="Needs Metadata", metadata={})
            _seed_document(
                db,
                owner="alice",
                title="Already Reindexed",
                metadata={
                    "section_key": "bai-thuc-hanh-so-4",
                    "section_title": "Bài thực hành số 4",
                    "char_start": 0,
                    "char_end": 42,
                },
            )
            _seed_document(db, owner="bob", title="Other Owner", metadata={})

            summary = run_backfill(db, owner_username="alice", apply=False)

            assert summary["selected_documents"] == 2
            statuses = {item["title"]: item["status"] for item in summary["results"]}
            assert statuses == {
                "Needs Metadata": "dry_run",
                "Already Reindexed": "skipped_up_to_date",
            }
            assert {item["owner_username"] for item in summary["results"]} == {"alice"}
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)
