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


def _settings(**overrides):
    defaults = {
        "retrieval_top_k": 1,
        "retrieval_max_rerank_candidates": 20,
        "retrieval_min_score": 0.0,
        "retrieval_min_lexical_score": 0.0,
        "retrieval_hybrid_semantic_weight": 0.5,
        "retrieval_hybrid_lexical_weight": 0.5,
        "retrieval_rerank_title_weight": 0.1,
        "retrieval_rerank_position_weight": 0.05,
        "retrieval_low_confidence_score": 0.15,
        "retrieval_enable_query_expansion": False,
        "retrieval_max_context_chunks": 3,
        "vector_store_provider": "database",
        "vector_store_collection": "knowledge_chunks",
    }
    defaults.update(overrides)
    mock = MagicMock()
    for key, value in defaults.items():
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


def _provider(chunk_count: int):
    from app.services.embeddings.base import EmbedResult, EmbeddingMeta, QueryEmbedResult

    meta = EmbeddingMeta(provider="api", model="test-embed", dimensions=3, version="test-v1", extra={})
    provider = MagicMock()
    provider.embed_query.return_value = QueryEmbedResult(vector=[1.0, 0.0, 0.0], meta=meta)
    provider.embed_texts.return_value = EmbedResult(vectors=[[1.0, 0.0, 0.0] for _ in range(chunk_count)], meta=meta)
    return provider


def _seed_doc(db, *, owner="alice", session_id=77, count=5):
    doc = KnowledgeDocument(owner_username=owner, title="Adjacent Doc", source_type="text", raw_text="standard retrieval", status="indexed", session_id=session_id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    for idx in range(count):
        db.add(KnowledgeChunk(
            document_id=doc.id,
            chunk_index=idx,
            content=f"standard retrieval context chunk {idx}",
            token_count=8,
            metadata_json={"embedding_provider": "api", "embedding_model": "test-embed", "embedding_dimensions": 3, "embedding_version": "test-v1"},
        ))
    other = KnowledgeDocument(owner_username=owner, title="Other Session", source_type="text", raw_text="other", status="indexed", session_id=99)
    db.add(other)
    db.commit()
    db.refresh(other)
    db.add(KnowledgeChunk(document_id=other.id, chunk_index=1, content="other session chunk must not appear", token_count=8, metadata_json={"embedding_provider": "api", "embedding_model": "test-embed"}))
    db.commit()
    return int(doc.id)


def test_standard_retrieval_adds_adjacent_chunks_with_scope_filters():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            doc_id = _seed_doc(db)
            provider = _provider(chunk_count=5)
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=True), \
                 patch("app.services.retrieval_service.vector_store.search_similar_chunks", return_value=[{"chunk_id": 4, "score": 0.95}]), \
                 patch("app.services.retrieval_service.settings", _settings()):
                result = search_knowledge(db, "alice", "standard retrieval", top_k=1, session_id=77)
            assert result["rag_mode"] == "session_rag"
            assert result["adjacent_expansion_applied"] is True
            assert result["adjacent_expanded_count"] == 2
            assert [row["chunk_index"] for row in result["results"]] == [2, 3, 4]
            assert {row["document_id"] for row in result["results"]} == {doc_id}
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_section_retrieval_skips_adjacent_expansion():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            doc = KnowledgeDocument(owner_username="alice", title="Practice", source_type="text", raw_text="Bài thực hành số 4", status="indexed", session_id=44)
            db.add(doc)
            db.commit()
            db.refresh(doc)
            for idx in range(3):
                db.add(KnowledgeChunk(document_id=doc.id, chunk_index=idx, content=f"Bài thực hành số 4 item {idx}", token_count=8, metadata_json={"section_key": "bai-thuc-hanh-so-4", "section_title": "Bài thực hành số 4", "section_order": 0}))
            db.commit()
            with patch("app.services.retrieval_service.crud_knowledge.get_adjacent_chunks") as adjacent, \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=True), \
                 patch("app.services.retrieval_service.get_embedding_provider") as embed_provider, \
                 patch("app.services.retrieval_service.settings", _settings()):
                result = search_knowledge(db, "alice", "Bài thực hành số 4 có mấy bài, tóm tắt từng bài", session_id=44)
            assert result["rag_mode"] == "section_rag"
            assert "adjacent_expansion_applied" not in result
            adjacent.assert_not_called()
            embed_provider.assert_not_called()
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)
