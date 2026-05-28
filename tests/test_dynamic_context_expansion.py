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
        "retrieval_top_k": 2,
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


def _seed_doc(db, *, owner="alice", session_id=55, count=5, section_metadata=False):
    doc = KnowledgeDocument(owner_username=owner, title="Context Doc", source_type="text", raw_text="context", status="indexed", session_id=session_id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    for idx in range(count):
        metadata = {"embedding_provider": "api", "embedding_model": "test-embed", "embedding_dimensions": 3, "embedding_version": "test-v1"}
        if section_metadata:
            metadata.update({"section_key": "bai-thuc-hanh-so-4", "section_title": "Bài thực hành số 4", "section_order": 0})
        db.add(KnowledgeChunk(document_id=doc.id, chunk_index=idx, content=f"context count summary chunk {idx}", token_count=8, metadata_json=metadata))
    db.commit()
    return int(doc.id)


def test_count_summary_query_uses_capped_dynamic_context_when_no_section_match():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            _seed_doc(db, count=5, section_metadata=False)
            provider = _provider(chunk_count=5)
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=False), \
                 patch("app.services.retrieval_service.settings", _settings(retrieval_top_k=2, retrieval_max_context_chunks=3)):
                result = search_knowledge(db, "alice", "có mấy nội dung, tóm tắt từng mục", session_id=55)
            assert result["rag_mode"] == "session_rag"
            assert result["dynamic_context_expanded"] is True
            assert result["requested_top_k"] == 2
            assert result["effective_top_k"] == 3
            assert result["returned"] <= 3
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_non_expansive_query_keeps_default_retrieval_depth():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            _seed_doc(db, count=5, section_metadata=False)
            provider = _provider(chunk_count=5)
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=False), \
                 patch("app.services.retrieval_service.settings", _settings(retrieval_top_k=2, retrieval_max_context_chunks=3)):
                result = search_knowledge(db, "alice", "context chunk 1", session_id=55)
            assert result["dynamic_context_expanded"] is False
            assert result["requested_top_k"] == 2
            assert result["effective_top_k"] == 2
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_high_confidence_section_query_does_not_use_dynamic_expansion():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _factory()
    try:
        db = factory()
        try:
            _seed_doc(db, count=3, section_metadata=True)
            with patch("app.services.retrieval_service.get_embedding_provider") as provider, \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=False), \
                 patch("app.services.retrieval_service.settings", _settings(retrieval_top_k=2, retrieval_max_context_chunks=3)):
                result = search_knowledge(db, "alice", "Bài thực hành số 4 có mấy bài, tóm tắt từng bài", session_id=55)
            assert result["rag_mode"] == "section_rag"
            assert "dynamic_context_expanded" not in result
            provider.assert_not_called()
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)
