from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.knowledge_models import KnowledgeChunk, KnowledgeDocument


def _settings():
    mock = MagicMock()
    for key, value in {
        "retrieval_top_k": 5,
        "retrieval_max_rerank_candidates": 20,
        "retrieval_min_score": 0.0,
        "retrieval_min_lexical_score": 0.0,
        "retrieval_hybrid_semantic_weight": 0.5,
        "retrieval_hybrid_lexical_weight": 0.5,
        "retrieval_rerank_title_weight": 0.1,
        "retrieval_rerank_position_weight": 0.05,
        "retrieval_low_confidence_score": 0.15,
        "retrieval_enable_query_expansion": False,
        "vector_store_provider": "database",
        "vector_store_collection": "knowledge_chunks",
    }.items():
        setattr(mock, key, value)
    return mock


def _factory():
    import app.models.chat_models  # noqa: F401
    import app.models.system_models  # noqa: F401

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return session_factory, engine, db_path


def _seed_section_doc(db, owner="alice", session_id=44):
    doc = KnowledgeDocument(owner_username=owner, title="Synthetic Practice", source_type="text", raw_text="Bài thực hành số 4", status="indexed", session_id=session_id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    contents = ["Bài thực hành số 4\n1. Nội dung bài một.", "2. Nội dung bài hai.\n3. Nội dung bài ba."]
    for idx, content in enumerate(contents):
        db.add(KnowledgeChunk(document_id=doc.id, chunk_index=idx, content=content, token_count=8, metadata_json={"section_key": "bai-thuc-hanh-so-4", "section_title": "Bài thực hành số 4", "section_level": 2, "section_order": 0, "page_number": 1, "char_start": idx * 50}))
    db.commit()
    return int(doc.id)


def test_section_retrieval_mvp_runs_before_vector_search_for_synthetic_practice_doc():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            doc_id = _seed_section_doc(db, owner="alice", session_id=44)
            with patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=True), \
                 patch("app.services.retrieval_service.vector_store.search_similar_chunks") as vector_search, \
                 patch("app.services.retrieval_service.get_embedding_provider") as embed_provider, \
                 patch("app.services.retrieval_service.settings", _settings()):
                result = search_knowledge(db, "alice", "Bài thực hành số 4 có mấy bài, tóm tắt từng bài", session_id=44)
            assert result["rag_mode"] == "section_rag"
            assert result["section_key"] == "bai-thuc-hanh-so-4"
            assert result["vector_store_attempted"] is False
            assert [row["chunk_index"] for row in result["results"]] == [0, 1]
            assert {row["document_id"] for row in result["results"]} == {doc_id}
            assert "1. Nội dung bài một" in "\n".join(row["content"] for row in result["results"])
            assert "3. Nội dung bài ba" in "\n".join(row["content"] for row in result["results"])
            vector_search.assert_not_called()
            embed_provider.assert_not_called()
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)

