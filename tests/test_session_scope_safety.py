from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.knowledge_models import KnowledgeChunk, KnowledgeDocument


def _make_settings_override(**kwargs):
    defaults = {
        "embedding_provider": "local",
        "embedding_model": "local-hash-v1",
        "embedding_dimensions": 3,
        "embedding_base_url": "http://localhost:11434",
        "embedding_timeout_seconds": 60.0,
        "embedding_batch_size": 16,
        "ingestion_pipeline": "custom",
        "vector_store_provider": "database",
        "vector_store_collection": "knowledge_chunks",
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
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for key, value in defaults.items():
        setattr(mock, key, value)
    return mock


def _make_sqlite_factory():
    import app.models.chat_models  # noqa: F401
    import app.models.system_models  # noqa: F401

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return session_factory, engine, db_path


def _seed_document(
    session_factory,
    *,
    owner: str = "alice",
    title: str = "Test Doc",
    status: str = "indexed",
    session_id: int | None = None,
) -> int:
    db = session_factory()
    try:
        doc = KnowledgeDocument(
            owner_username=owner,
            title=title,
            source_type="text",
            raw_text=f"{title} raw text",
            status=status,
            session_id=session_id,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return int(doc.id)
    finally:
        db.close()


def _seed_chunk(session_factory, document_id: int, *, content: str = "refund policy details") -> int:
    db = session_factory()
    try:
        chunk = KnowledgeChunk(
            document_id=document_id,
            chunk_index=0,
            content=content,
            token_count=3,
            embedding_model="local-hash-v1",
            vector_id=f"vec-{document_id}-0",
            metadata_json={
                "embedding": [0.2, 0.2, 0.2],
                "embedding_provider": "local",
                "embedding_model": "local-hash-v1",
                "embedding_dimensions": 3,
                "embedding_version": "local-hash-v1",
            },
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)
        return int(chunk.id)
    finally:
        db.close()


def _mock_provider():
    from app.services.embeddings.base import EmbeddingMeta, QueryEmbedResult

    meta = EmbeddingMeta(
        provider="local",
        model="local-hash-v1",
        dimensions=3,
        version="local-hash-v1",
    )
    provider = MagicMock()
    provider.embed_query.return_value = QueryEmbedResult(vector=[0.2, 0.2, 0.2], meta=meta)
    return provider


def test_session_with_only_non_indexed_documents_is_indexing_pending_without_embedding():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _make_sqlite_factory()
    try:
        owner = "session_pending_user"
        _seed_document(factory, owner=owner, title="Pending Session Doc", status="uploaded", session_id=101)
        global_doc_id = _seed_document(factory, owner=owner, title="Global Indexed Doc", status="indexed")
        _seed_chunk(factory, global_doc_id, content="global refund policy details")
        provider = _mock_provider()

        db = factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=False), \
                 patch("app.services.retrieval_service.settings", _make_settings_override()):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="refund policy",
                    session_id=101,
                    request_id="req-indexing-pending",
                )

            assert result["returned"] == 0
            assert result["results"] == []
            assert result["rag_mode"] == "indexing_pending"
            assert result["retrieval_scope"] == "session"
            assert result["fallback_reason"] == "no_indexed_session_documents"
            assert result["vector_store_attempted"] is False
            provider.embed_query.assert_not_called()
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_indexed_session_retrieval_does_not_return_global_documents():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _make_sqlite_factory()
    try:
        owner = "session_scope_user"
        session_doc_id = _seed_document(factory, owner=owner, title="Session Indexed Doc", status="indexed", session_id=202)
        global_doc_id = _seed_document(factory, owner=owner, title="Global Indexed Doc", status="indexed")
        _seed_chunk(factory, session_doc_id, content="session-only refund policy details")
        _seed_chunk(factory, global_doc_id, content="global refund policy details")
        provider = _mock_provider()

        db = factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=False), \
                 patch("app.services.retrieval_service.settings", _make_settings_override()):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="refund policy",
                    session_id=202,
                    request_id="req-session-scope",
                )

            returned_doc_ids = {item["document_id"] for item in result["results"]}
            assert result["retrieval_scope"] == "session"
            assert result["rag_mode"] == "session_rag"
            assert session_doc_id in returned_doc_ids
            assert global_doc_id not in returned_doc_ids
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_explicit_document_scope_preserves_selected_document_metadata():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _make_sqlite_factory()
    try:
        owner = "document_scope_user"
        doc_id = _seed_document(factory, owner=owner, title="Selected Doc", status="indexed")
        _seed_chunk(factory, doc_id, content="selected document refund policy details")
        provider = _mock_provider()

        db = factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=False), \
                 patch("app.services.retrieval_service.settings", _make_settings_override()):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="refund policy",
                    document_id=doc_id,
                    session_id=303,
                    request_id="req-document-scope",
                )

            assert result["retrieval_scope"] == "document"
            assert result["rag_mode"] == "document_rag"
            assert result["selected_document_id"] == doc_id
            assert result["session_id"] == 303
            assert {item["document_id"] for item in result["results"]} == {doc_id}
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_vector_store_failure_sets_safe_metadata_and_uses_db_fallback():
    from app.services.retrieval_service import search_knowledge

    factory, engine, db_path = _make_sqlite_factory()
    try:
        owner = "vector_failure_user"
        doc_id = _seed_document(factory, owner=owner, title="Vector Failure Doc", status="indexed")
        _seed_chunk(factory, doc_id, content="refund policy details")
        provider = _mock_provider()

        db = factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider", return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled", return_value=True), \
                 patch(
                     "app.services.retrieval_service.vector_store.search_similar_chunks",
                     side_effect=RuntimeError("secret-url password=hidden"),
                 ), \
                 patch("app.services.retrieval_service.settings", _make_settings_override(vector_store_provider="qdrant")):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="refund policy",
                    document_id=doc_id,
                    request_id="req-vector-failure",
                )

            assert result["returned"] == 1
            assert result["vector_store_attempted"] is True
            assert result["vector_store_failed"] is True
            assert result["vector_store_error_type"] == "RuntimeError"
            assert result["fallback_reason"] == "vector_store_failure"
            assert "secret-url" not in result["vector_store_error_type"]
            assert "password" not in result["vector_store_error_type"]
        finally:
            db.close()
    finally:
        engine.dispose()
        os.remove(db_path)


def test_prepare_chat_turn_with_non_indexed_session_docs_does_not_call_search():
    from app.services import chat_service

    db = MagicMock()
    user = SimpleNamespace(username="alice", max_tokens_per_day=10000)
    session = SimpleNamespace(id=42, username="alice", title="Pending docs")
    pending_doc = SimpleNamespace(id=7, title="Pending", session_id=42, status="uploaded")
    user_msg = SimpleNamespace(id=1001)
    model_selection = SimpleNamespace(context_window=4096, max_output_tokens=512)
    registry = MagicMock()
    registry.select_model.return_value = model_selection

    with patch.object(chat_service.crud_chat, "get_user_by_username", return_value=user), \
         patch.object(chat_service.crud_chat, "get_chat_session", return_value=session), \
         patch.object(chat_service.crud_chat, "count_session_messages", return_value=0), \
         patch.object(chat_service.crud_chat, "get_rolling_token_usage", return_value={"total_tokens": 0}), \
         patch.object(chat_service.crud_chat, "create_message", return_value=user_msg), \
         patch.object(chat_service.crud_knowledge, "list_documents", side_effect=[[pending_doc], []]), \
         patch.object(chat_service, "_build_hybrid_context", return_value=(None, [])), \
         patch.object(chat_service.llm_provider, "get_provider_registry", return_value=registry), \
         patch.object(chat_service, "search_knowledge") as search_knowledge:
        prepared = chat_service._prepare_chat_turn(
            db,
            "alice",
            42,
            "summarize pending doc",
            knowledge_document_id=None,
            use_web_search=False,
        )

    search_knowledge.assert_not_called()
    assert prepared.knowledge_base_active is False
    assert prepared.retrieval_result["rag_mode"] == "indexing_pending"
    assert prepared.retrieval_result["retrieval_scope"] == "session"
    assert prepared.retrieval_result["fallback_reason"] == "no_indexed_session_documents"
    assert prepared.retrieval_result["vector_store_attempted"] is False
