from __future__ import annotations

import os
from types import SimpleNamespace

os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-rag-core-api-mode"


def test_prepare_chunks_for_indexing_delegates_to_rag_core_api(monkeypatch):
    from app.services import knowledge_service

    class FakeClient:
        def prepare_indexing(self, **kwargs):
            assert kwargs["document_id"] == 7
            assert kwargs["index_provider"] == "qdrant"
            return {"prepared_chunks": [{"chunk_index": 0, "content": "prepared"}]}

    monkeypatch.setattr(knowledge_service, "is_rag_core_api_mode", lambda: True)
    monkeypatch.setattr(knowledge_service, "get_rag_core_client", lambda: FakeClient())
    monkeypatch.setattr(knowledge_service.vector_store, "should_store_embeddings_in_database", lambda: False)
    monkeypatch.setattr(knowledge_service.settings, "vector_store_provider", "qdrant")

    prepared = knowledge_service.prepare_chunks_for_indexing(
        7,
        "abcdef123456",
        [{"chunk_index": 0, "content": "raw", "metadata_json": {}}],
    )
    assert prepared == [{"chunk_index": 0, "content": "prepared"}]


def test_vector_store_delegates_to_rag_core_api(monkeypatch):
    from app.services import vector_store

    calls: list[tuple[str, dict]] = []

    class FakeClient:
        def vector_delete(self, **kwargs):
            calls.append(("delete", kwargs))
            return {"ok": True, "deleted": True}

        def vector_upsert(self, **kwargs):
            calls.append(("upsert", kwargs))
            return {"ok": True, "upserted": True}

        def vector_search(self, **kwargs):
            calls.append(("search", kwargs))
            return {"vector_hits": [{"chunk_id": 1, "document_id": 2, "score": 0.8, "vector_id": "1"}]}

    monkeypatch.setattr(vector_store, "is_rag_core_api_mode", lambda: True)
    monkeypatch.setattr(vector_store, "is_external_vector_store_enabled", lambda: True)
    monkeypatch.setattr(vector_store, "get_rag_core_client", lambda: FakeClient())

    vector_store.delete_document_chunks("alice", 2)
    doc = SimpleNamespace(
        id=2,
        owner_username="alice",
        title="Doc",
        source_type="text",
        source_uri=None,
        session_id=None,
    )
    row = SimpleNamespace(id=1, chunk_index=0)
    vector_store.upsert_document_chunks(
        doc,
        [row],
        [{"chunk_index": 0, "embedding": [0.1], "metadata_json": {"embedding_provider": "local"}}],
    )
    hits = vector_store.search_similar_chunks("alice", [0.1], top_k=1)

    assert calls[0][0] == "delete"
    assert calls[1][0] == "upsert"
    assert calls[2][0] == "search"
    assert hits[0]["chunk_id"] == 1


def test_rag_core_api_unavailable_result_is_not_grounded(monkeypatch):
    from app.services import retrieval_service

    monkeypatch.setattr(
        retrieval_service.crud_knowledge,
        "create_retrieval_event",
        lambda *args, **kwargs: SimpleNamespace(id=123),
    )

    result = retrieval_service._rag_core_api_unavailable_result(
        SimpleNamespace(),
        owner_username="alice",
        query="question",
        top_k=5,
        request_id="req-1",
        started_at=0.0,
        document_id=None,
        session_id=44,
        session_scope="session",
        retrieval_scope="session",
        rag_mode="session_rag",
        error_type="ConnectError",
    )

    assert result["returned"] == 0
    assert result["results"] == []
    assert result["evidence_strength"] == "none"
    assert result["fallback_reason"] == "rag_core_api_unavailable"
    assert result["vector_store_failed"] is True
    assert result["vector_store_error_type"] == "ConnectError"

