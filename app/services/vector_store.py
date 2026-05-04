from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def provider_name() -> str:
    return (settings.vector_store_provider or "database").strip().lower()


def is_external_vector_store_enabled() -> bool:
    return provider_name() == "qdrant"


def should_store_embeddings_in_database() -> bool:
    return not is_external_vector_store_enabled()


@lru_cache
def _get_qdrant_client():
    if not is_external_vector_store_enabled():
        return None
    if not (settings.vector_store_url or "").strip():
        raise RuntimeError("VECTOR_STORE_URL is required when VECTOR_STORE_PROVIDER=qdrant.")

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for Qdrant vector store support.") from exc

    return QdrantClient(
        url=settings.vector_store_url,
        api_key=settings.vector_store_api_key or None,
        timeout=settings.vector_store_timeout_seconds,
        prefer_grpc=settings.vector_store_prefer_grpc,
    )


def check_vector_store_health() -> dict:
    started_at = perf_counter()
    provider = provider_name()
    dependency = "qdrant" if provider == "qdrant" else "vector_store"
    base = {
        "ok": False,
        "dependency": dependency,
        "provider": provider,
        "collection": settings.vector_store_collection,
        "url": settings.vector_store_url,
    }

    if not is_external_vector_store_enabled():
        return {
            **base,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "detail": "External vector store is disabled.",
        }

    try:
        client = _get_qdrant_client()
        collections_response = client.get_collections()
        collection_names = [item.name for item in collections_response.collections]
        return {
            **base,
            "ok": True,
            "collection_exists": settings.vector_store_collection in collection_names,
            "collections_count": len(collection_names),
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }
    except Exception as exc:
        return {
            **base,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "detail": str(exc),
        }


def _ensure_qdrant_collection(vector_size: int) -> None:
    client = _get_qdrant_client()
    if client is None:
        return

    from qdrant_client import models
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client.get_collection(settings.vector_store_collection)
        return
    except (UnexpectedResponse, ValueError):
        pass

    client.create_collection(
        collection_name=settings.vector_store_collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def delete_document_chunks(owner_username: str, document_id: int) -> None:
    if not is_external_vector_store_enabled():
        return

    client = _get_qdrant_client()
    from qdrant_client import models
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client.delete(
            collection_name=settings.vector_store_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="owner_username",
                            match=models.MatchValue(value=owner_username),
                        ),
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )
    except (UnexpectedResponse, ValueError) as exc:
        logger.info("Skipping Qdrant delete for missing collection or filter support: %s", exc)


def upsert_document_chunks(document, chunk_rows: list, prepared_chunks: list[dict]) -> None:
    if not is_external_vector_store_enabled() or not chunk_rows:
        return

    embeddings_by_index = {
        int(chunk["chunk_index"]): chunk.get("embedding") or []
        for chunk in prepared_chunks
    }
    first_vector = next((vector for vector in embeddings_by_index.values() if vector), None)
    if not first_vector:
        raise ValueError("No embeddings prepared for vector upsert.")

    _ensure_qdrant_collection(len(first_vector))
    client = _get_qdrant_client()
    from qdrant_client import models

    session_scope = "session" if getattr(document, "session_id", None) else "global"
    points = []
    for row in chunk_rows:
        vector = embeddings_by_index.get(int(row.chunk_index)) or []
        if not vector:
            continue
        points.append(
            models.PointStruct(
                id=int(row.id),
                vector=vector,
                payload={
                    "owner_username": document.owner_username,
                    "document_id": int(document.id),
                    "chunk_id": int(row.id),
                    "chunk_index": int(row.chunk_index),
                    "title": document.title,
                    "source_type": document.source_type,
                    "source_uri": document.source_uri,
                    "session_id": int(document.session_id) if document.session_id is not None else None,
                    "session_scope": session_scope,
                },
            )
        )

    if not points:
        logger.warning("No Qdrant points built for document id=%s", document.id)
        return

    client.upsert(
        collection_name=settings.vector_store_collection,
        wait=True,
        points=points,
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

    client = _get_qdrant_client()
    from qdrant_client import models

    must_conditions = [
        models.FieldCondition(
            key="owner_username",
            match=models.MatchValue(value=owner_username),
        )
    ]

    if document_id is not None:
        must_conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )
    elif session_scope == "session" and session_id is not None:
        must_conditions.extend(
            [
                models.FieldCondition(
                    key="session_scope",
                    match=models.MatchValue(value="session"),
                ),
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                ),
            ]
        )
    elif session_scope == "global":
        must_conditions.append(
            models.FieldCondition(
                key="session_scope",
                match=models.MatchValue(value="global"),
            )
        )

    points = client.search(
        collection_name=settings.vector_store_collection,
        query_vector=query_vector,
        query_filter=models.Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "chunk_id": int(point.payload.get("chunk_id") or point.id),
            "document_id": int(point.payload.get("document_id")),
            "score": float(point.score or 0.0),
            "vector_id": str(point.id),
        }
        for point in points
    ]