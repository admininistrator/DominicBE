from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"


def _settings(**overrides):
    values = {
        "retrieval_top_k": 5,
        "retrieval_max_rerank_candidates": 20,
        "retrieval_min_score": 0.0,
        "retrieval_min_lexical_score": 0.0,
        "retrieval_hybrid_semantic_weight": 0.5,
        "retrieval_hybrid_lexical_weight": 0.5,
        "retrieval_enable_query_expansion": False,
        "retrieval_rerank_title_weight": 0.1,
        "retrieval_rerank_position_weight": 0.05,
        "retrieval_low_confidence_score": 0.15,
        "vector_store_provider": "qdrant",
        "vector_store_collection": "knowledge_chunks",
    }
    values.update(overrides)
    mock = MagicMock()
    for key, value in values.items():
        setattr(mock, key, value)
    return mock


def _query_embed_result(vector=None):
    from app.services.embeddings.base import EmbeddingMeta, QueryEmbedResult

    return QueryEmbedResult(
        vector=vector or [0.1, 0.2, 0.3],
        meta=EmbeddingMeta(
            provider="local",
            model="local-hash-v1",
            dimensions=3,
            version="local-hash-v1",
            extra={},
        ),
    )


def _point(point_id=11, score=0.88, payload=None):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload=payload
        or {
            "chunk_id": point_id,
            "document_id": 7,
        },
    )


def test_qdrant_adapter_uses_query_points_and_preserves_filters():
    from rag_core.vector_store.qdrant_adapter import QdrantAdapter

    adapter = QdrantAdapter(
        collection="knowledge_chunks",
        url="http://localhost:6333",
    )
    mock_client = MagicMock()
    mock_client.query_points.return_value = SimpleNamespace(points=[_point()])
    mock_client.search.side_effect = AssertionError("search should not be called")

    with patch.object(adapter, "_get_client", return_value=mock_client):
        results = adapter.search_similar_chunks(
            "alice",
            [0.1, 0.2, 0.3],
            top_k=3,
            session_id=42,
            session_scope="session",
        )

    assert results == [
        {
            "chunk_id": 11,
            "document_id": 7,
            "score": 0.88,
            "vector_id": "11",
        }
    ]
    mock_client.search.assert_not_called()
    mock_client.query_points.assert_called_once()
    kwargs = mock_client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "knowledge_chunks"
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] == 3
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False

    conditions = {
        condition.key: condition.match.value
        for condition in kwargs["query_filter"].must
    }
    assert conditions == {
        "owner_username": "alice",
        "session_scope": "session",
        "session_id": 42,
    }


def test_qdrant_adapter_normalizes_points_attribute_and_list_results():
    from rag_core.vector_store.qdrant_adapter import QdrantAdapter

    point = _point()
    assert QdrantAdapter._normalize_query_points_result(
        SimpleNamespace(points=[point])
    ) == [point]
    assert QdrantAdapter._normalize_query_points_result([point]) == [point]


def test_qdrant_adapter_document_filter_is_passed():
    from rag_core.vector_store.qdrant_adapter import QdrantAdapter

    adapter = QdrantAdapter(
        collection="knowledge_chunks",
        url="http://localhost:6333",
    )
    mock_client = MagicMock()
    mock_client.query_points.return_value = [_point()]

    with patch.object(adapter, "_get_client", return_value=mock_client):
        adapter.search_similar_chunks(
            "alice",
            [0.1],
            top_k=1,
            document_id=99,
        )

    kwargs = mock_client.query_points.call_args.kwargs
    conditions = {
        condition.key: condition.match.value
        for condition in kwargs["query_filter"].must
    }
    assert conditions == {
        "owner_username": "alice",
        "document_id": 99,
    }


@patch("app.services.retrieval_service.settings", new_callable=lambda: _settings())
@patch("app.services.retrieval_service.get_embedding_provider")
@patch("app.services.retrieval_service.vector_store")
@patch("app.services.retrieval_service.crud_knowledge")
def test_search_knowledge_falls_back_when_qdrant_raises(
    mock_crud,
    mock_vector_store,
    mock_get_provider,
    mock_settings,
):
    from app.services.retrieval_service import search_knowledge

    provider = MagicMock()
    provider.embed_query.return_value = _query_embed_result()
    mock_get_provider.return_value = provider

    mock_vector_store.is_external_vector_store_enabled.return_value = True
    mock_vector_store.search_similar_chunks.side_effect = RuntimeError(
        "backend failure"
    )

    chunk = SimpleNamespace(
        id=11,
        chunk_index=0,
        content="refund policy details",
        token_count=3,
        vector_id="11",
        embedding_model="local-hash-v1",
        metadata_json={
            "embedding": [0.1, 0.2, 0.3],
            "embedding_provider": "local",
            "embedding_model": "local-hash-v1",
        },
    )
    document = SimpleNamespace(
        id=7,
        title="Policy",
        source_type="upload",
        source_uri="policy.pdf",
    )
    mock_crud.list_searchable_chunks.return_value = [(chunk, document)]
    mock_event = SimpleNamespace(id=123)
    mock_crud.create_retrieval_event.return_value = mock_event

    with patch("app.services.retrieval_service.logger") as mock_logger:
        result = search_knowledge(
            db=MagicMock(),
            owner_username="alice",
            query="refund policy",
            top_k=1,
        )

    assert result["returned"] == 1
    assert result["results"][0]["chunk_id"] == 11
    mock_crud.list_searchable_chunks.assert_called_once()
    mock_logger.warning.assert_called_once()
    log_args = mock_logger.warning.call_args.args
    assert "error_type=%s collection=%s top_k=%s" in log_args[0]
    assert log_args[1:] == ("RuntimeError", "knowledge_chunks", 20)
