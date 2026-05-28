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


def _factory():
    import app.models.chat_models  # noqa: F401
    import app.models.system_models  # noqa: F401

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return session_factory, engine, db_path


def _seed(db, owner="alice", session_id=10, section_key="bai-thuc-hanh-so-4"):
    doc = KnowledgeDocument(owner_username=owner, title="Doc", source_type="text", raw_text="raw", status="indexed", session_id=session_id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    for idx in range(2):
        db.add(KnowledgeChunk(document_id=doc.id, chunk_index=idx, content=f"chunk {idx}", token_count=2, metadata_json={"section_key": section_key, "section_title": "Bài thực hành số 4", "section_order": 0, "char_start": idx * 10}))
    db.commit()
    return doc.id


def test_find_chunks_by_section_key_preserves_owner_session_filters_and_order():
    from app.crud import crud_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            doc_id = _seed(db, owner="alice", session_id=10)
            _seed(db, owner="bob", session_id=10)
            _seed(db, owner="alice", session_id=99)
            rows = crud_knowledge.find_chunks_by_section_key(db, "alice", "bai-thuc-hanh-so-4", session_id=10, session_scope="session")
            assert [doc.id for _chunk, doc in rows] == [doc_id, doc_id]
            assert [chunk.chunk_index for chunk, _doc in rows] == [0, 1]
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)

