"""Vector store service — delegates to ``rag_core.vector_store``.

WARNING: This module now wraps ``rag_core.vector_store.qdrant_adapter.QdrantAdapter``.
Do NOT edit Qdrant-specific logic here — modify ``rag_core.vector_store.qdrant_adapter`` instead.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag_core_client import get_rag_core_client, is_rag_core_api_mode
from rag_core.vector_store.qdrant_adapter import QdrantAdapter

logger = get_logger(__name__)


@lru_cache
def _get_adapter() -> QdrantAdapter:
    """Return a cached QdrantAdapter instance configured from settings."""
    return QdrantAdapter(
        collection=settings.vector_store_collection,
        url=settings.vector_store_url,
        api_key=settings.vector_store_api_key,
        timeout_seconds=settings.vector_store_timeout_seconds,
        prefer_grpc=settings.vector_store_prefer_grpc,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )


def provider_name() -> str:
    return (settings.vector_store_provider or "database").strip().lower()


def is_external_vector_store_enabled() -> bool:
    return provider_name() == "qdrant"


def should_store_embeddings_in_database() -> bool:
    return not is_external_vector_store_enabled()


def check_vector_store_health() -> dict:
    """Return health check information delegating to the QdrantAdapter."""
    if is_rag_core_api_mode():
        try:
            health = get_rag_core_client().health()
            dependencies = health.get("dependencies") if isinstance(health, dict) else None
            qdrant = (dependencies or {}).get("qdrant") if isinstance(dependencies, dict) else None
            if isinstance(qdrant, dict):
                return {**qdrant, "via": "rag-core-api"}
            return {
                "ok": bool(health.get("ok")) if isinstance(health, dict) else False,
                "provider": "qdrant",
                "via": "rag-core-api",
                "detail": "rag-core health did not include qdrant dependency",
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "qdrant",
                "via": "rag-core-api",
                "detail": type(exc).__name__,
            }
    if not is_external_vector_store_enabled():
        return _get_adapter().check_health()
    return _get_adapter().check_health()


def delete_document_chunks(owner_username: str, document_id: int) -> None:
    if not is_external_vector_store_enabled():
        return
    if is_rag_core_api_mode():
        get_rag_core_client().vector_delete(
            owner_username=owner_username,
            document_id=document_id,
        )
        return
    _get_adapter().delete_document_chunks(owner_username, document_id)


def upsert_document_chunks(document, chunk_rows: list, prepared_chunks: list[dict]) -> None:
    if not is_external_vector_store_enabled() or not chunk_rows:
        return

    # Build provider metadata from the first prepared chunk
    embedding_provider = settings.embedding_provider
    embedding_model = settings.embedding_model
    meta_by_index = {
        int(chunk["chunk_index"]): (chunk.get("metadata_json") or {})
        for chunk in prepared_chunks
    }
    if meta_by_index:
        first_meta = next(iter(meta_by_index.values()))
        embedding_provider = first_meta.get("embedding_provider", embedding_provider)
        embedding_model = first_meta.get("embedding_model", embedding_model)

    if is_rag_core_api_mode():
        get_rag_core_client().vector_upsert(
            document={
                "owner_username": document.owner_username,
                "document_id": int(document.id),
                "title": document.title,
                "source_type": document.source_type,
                "source_uri": document.source_uri,
                "session_id": int(document.session_id) if document.session_id is not None else None,
            },
            chunk_rows=[
                {
                    "id": int(row.id),
                    "chunk_index": int(row.chunk_index),
                }
                for row in chunk_rows
            ],
            prepared_chunks=prepared_chunks,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        return

    adapter = _get_adapter()
    adapter.upsert_document_chunks(
        owner_username=document.owner_username,
        document_id=int(document.id),
        title=document.title,
        source_type=document.source_type,
        source_uri=document.source_uri,
        session_id=int(document.session_id) if document.session_id is not None else None,
        chunk_rows=chunk_rows,
        prepared_chunks=prepared_chunks,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


def search_similar_chunks(
    owner_username: str,
    query_vector: list[float],
    *,
    top_k: int,
    document_id: int | None = None,
    session_id: int | None = None,
    session_scope: str = "all",
) -> list[dict]:
    if not is_external_vector_store_enabled():
        return []
    if is_rag_core_api_mode():
        response = get_rag_core_client().vector_search(
            owner_username=owner_username,
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            session_id=session_id,
            session_scope=session_scope,
        )
        return response.get("vector_hits") or []
    return _get_adapter().search_similar_chunks(
        owner_username,
        query_vector,
        top_k=top_k,
        document_id=document_id,
        session_id=session_id,
        session_scope=session_scope,
    )
