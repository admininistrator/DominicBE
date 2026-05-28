"""Knowledge retrieval service for Phase 2 searchable indexing.

Delegates pure scoring/ranking/evidence/compat functions to ``rag_core.retrieval``.
Retains ``search_knowledge()`` as the orchestrator (DB queries, event logging,
fallback logic).
"""
from __future__ import annotations

import logging
import time
from math import sqrt
import re
from types import SimpleNamespace
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.json_utils import ensure_json_mapping
from app.crud import crud_knowledge
from app.services import vector_store
from app.services.embeddings.factory import get_embedding_provider
from app.services.rag_core_client import RagCoreClientError, get_rag_core_client, is_rag_core_api_mode

# rag-core imports for pure scoring/ranking/evidence functions
from rag_core.retrieval.query_processor import (
    QUERY_EXPANSION_RULES,  # noqa: F401
    _expand_query as _rag_expand_query,
    _normalize_for_search as _rag_normalize_for_search,
    _strip_accents as _rag_strip_accents,
    _tokenize as _rag_tokenize,
)
from rag_core.retrieval.scoring import (
    _cosine_similarity as _rag_cosine_similarity,
    _hybrid_score as _rag_hybrid_score,
    _lexical_overlap_score as _rag_lexical_overlap_score,
    _normalize_for_dedupe as _rag_normalize_for_dedupe,
)
from rag_core.retrieval.reranker import _rerank_results as _rag_rerank_results
from rag_core.retrieval.deduplicator import _dedupe_scored_results as _rag_dedupe_scored_results
from rag_core.retrieval.evidence import (
    _build_snippet as _rag_build_snippet,
    _classify_evidence_strength as _rag_classify_evidence_strength,
    _estimate_token_count as _rag_estimate_token_count,
    _is_embedding_compatible as _rag_is_embedding_compatible,
)
from rag_core.retrieval.section_retrieval import (
    SECTION_CONFIDENCE_THRESHOLD,
    detect_context_expansion_intent,
    match_section,
)

logger = get_logger(__name__)

RETRIEVAL_METADATA_DEFAULTS = {
    "rag_mode": "document_rag",
    "retrieval_scope": "global",
    "selected_document_id": None,
    "session_id": None,
    "section_key": None,
    "section_confidence": None,
    "vector_store_attempted": False,
    "vector_store_failed": False,
    "vector_store_error_type": None,
    "fallback_reason": None,
}


# ---------------------------------------------------------------------------
# Pure functions — thin wrappers that read settings and delegate to rag-core
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    return _rag_strip_accents(text)


def _normalize_for_search(text: str) -> str:
    return _rag_normalize_for_search(text)


def _tokenize(text: str) -> set[str]:
    return _rag_tokenize(text)


def _expand_query(query: str) -> tuple[str, list[str]]:
    return _rag_expand_query(query, enable_query_expansion=settings.retrieval_enable_query_expansion)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return _rag_cosine_similarity(left, right)


def _lexical_overlap_score(query_text: str, content: str) -> float:
    return _rag_lexical_overlap_score(query_text, content)


def _hybrid_score(semantic_score: float, lexical_score: float) -> float:
    return _rag_hybrid_score(
        semantic_score,
        lexical_score,
        semantic_weight=settings.retrieval_hybrid_semantic_weight,
        lexical_weight=settings.retrieval_hybrid_lexical_weight,
    )


def _normalize_for_dedupe(text: str) -> str:
    return _rag_normalize_for_dedupe(text)


def _rerank_results(query_text: str, results: list[dict]) -> list[dict]:
    return _rag_rerank_results(
        query_text,
        results,
        max_rerank_candidates=settings.retrieval_max_rerank_candidates,
        rerank_title_weight=settings.retrieval_rerank_title_weight,
        rerank_position_weight=settings.retrieval_rerank_position_weight,
    )


def _dedupe_scored_results(results: list[dict]) -> list[dict]:
    return _rag_dedupe_scored_results(results)


def _classify_evidence_strength(results: list[dict], *, fallback_used: bool) -> str:
    return _rag_classify_evidence_strength(
        results,
        fallback_used=fallback_used,
        low_confidence_score=settings.retrieval_low_confidence_score,
    )


def _build_snippet(content: str, *, max_chars: int = 220) -> str:
    return _rag_build_snippet(content, max_chars=max_chars)


def _estimate_token_count(text: str, explicit_count: int | None = None) -> int:
    return _rag_estimate_token_count(text, explicit_count=explicit_count)


def _is_embedding_compatible(chunk_meta: Any, query_provider: str, query_model: str) -> bool:
    return _rag_is_embedding_compatible(chunk_meta, query_provider, query_model)


def build_retrieval_metadata_contract(**overrides: Any) -> dict:
    """Build the additive retrieval metadata contract for chat/retrieval responses."""
    metadata = dict(RETRIEVAL_METADATA_DEFAULTS)
    metadata.update(overrides)
    return metadata


def _extract_chunk_metadata_fields(chunk_meta: dict) -> dict:
    """Return safe optional provenance metadata for a retrieval result."""
    fields = {}
    for key in (
        "page_number",
        "page_range",
        "section_key",
        "section_title",
        "section_level",
        "section_order",
        "char_start",
        "char_end",
    ):
        if key in chunk_meta:
            fields[key] = chunk_meta.get(key)
    return fields


def _resolve_retrieval_scope(document_id: int | None, session_id: int | None) -> tuple[str, str, str]:
    if document_id is not None:
        return "document", "document_rag", "all"
    if session_id is not None:
        return "session", "session_rag", "session"
    return "global", "document_rag", "all"


def _empty_retrieval_result(
    *,
    query: str,
    top_k: int,
    request_id: str | None,
    latency_ms: int,
    retrieval_id: int | None = None,
    document_id: int | None = None,
    session_id: int | None = None,
    session_scope: str = "all",
    strategy: str = "hybrid_rerank",
    fallback_reason: str | None = None,
    rag_mode: str = "indexing_pending",
    retrieval_scope: str = "session",
) -> dict:
    metadata = build_retrieval_metadata_contract(
        rag_mode=rag_mode,
        retrieval_scope=retrieval_scope,
        selected_document_id=document_id,
        session_id=session_id,
        fallback_reason=fallback_reason,
    )
    return {
        "query": query,
        "top_k": top_k,
        "returned": 0,
        "retrieval_id": retrieval_id,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "candidate_count": 0,
        "document_id": document_id,
        "strategy": strategy,
        "original_query": query,
        "rewritten_query": query,
        "query_expansions": [],
        "reranked_count": 0,
        "fallback_used": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "evidence_strength": "none",
        "session_scope": session_scope,
        "results": [],
        **metadata,
    }


# ---------------------------------------------------------------------------
# Stored embedding extraction
# ---------------------------------------------------------------------------

def _extract_stored_embedding(metadata_json: Any) -> list[float] | None:
    embedding = ensure_json_mapping(metadata_json).get("embedding")
    if isinstance(embedding, list) and embedding:
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError):
            pass
    return None


def _build_result_from_chunk(chunk, document, *, score: float, embed_provider: str | None = None) -> dict:
    chunk_meta = ensure_json_mapping(chunk.metadata_json)
    return {
        "document_id": document.id,
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "title": document.title,
        "source_type": document.source_type,
        "source_uri": document.source_uri,
        "score": score,
        "semantic_score": score,
        "lexical_score": score,
        "rerank_score": score,
        "token_count": int(chunk.token_count or 0) if getattr(chunk, "token_count", None) is not None else None,
        "token_estimate": _estimate_token_count(chunk.content, int(chunk.token_count or 0) if getattr(chunk, "token_count", None) is not None else None),
        "snippet": _build_snippet(chunk.content),
        "content": chunk.content,
        "vector_id": chunk.vector_id,
        "embedding_model": chunk.embedding_model,
        "embedding_provider": chunk_meta.get("embedding_provider", embed_provider),
        **_extract_chunk_metadata_fields(chunk_meta),
    }


def _embedding_meta_namespace(payload: dict | None = None):
    payload = payload or {}
    return SimpleNamespace(
        provider=payload.get("provider") or settings.embedding_provider,
        model=payload.get("model") or settings.embedding_model,
        dimensions=int(payload.get("dimensions") or settings.embedding_dimensions),
        version=payload.get("version") or "",
        extra=payload.get("extra") or {},
    )


def _serialize_retrieval_candidates(candidates: list[tuple]) -> list[dict]:
    serialized = []
    for chunk, document in candidates:
        serialized.append(
            {
                "chunk_id": int(chunk.id),
                "document_id": int(document.id),
                "chunk_index": int(chunk.chunk_index or 0),
                "title": document.title,
                "source_type": document.source_type,
                "source_uri": document.source_uri,
                "content": chunk.content,
                "token_count": int(chunk.token_count or 0) if getattr(chunk, "token_count", None) is not None else None,
                "vector_id": chunk.vector_id,
                "embedding_model": chunk.embedding_model,
                "metadata_json": ensure_json_mapping(chunk.metadata_json),
            }
        )
    return serialized


def _rag_core_api_unavailable_result(
    db: Session,
    *,
    owner_username: str,
    query: str,
    top_k: int,
    request_id: str | None,
    started_at: float,
    document_id: int | None,
    session_id: int | None,
    session_scope: str,
    retrieval_scope: str,
    rag_mode: str,
    error_type: str,
) -> dict:
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    fallback_reason = "rag_core_api_unavailable"
    retrieval_event = crud_knowledge.create_retrieval_event(
        db,
        username=owner_username,
        query_text=query,
        top_k=top_k,
        session_id=session_id,
        request_id=request_id,
        latency_ms=latency_ms,
        metadata_json={
            "document_id": document_id,
            "strategy": "rag_core_api",
            "original_query": query,
            "rewritten_query": query,
            "query_expansions": [],
            "returned": 0,
            "candidate_count": 0,
            "matched_count": 0,
            "reranked_count": 0,
            "fallback_used": False,
            "fallback_reason": fallback_reason,
            "evidence_strength": "none",
            "session_scope": session_scope,
            **build_retrieval_metadata_contract(
                rag_mode=rag_mode,
                retrieval_scope=retrieval_scope,
                selected_document_id=document_id,
                session_id=session_id,
                vector_store_attempted=False,
                vector_store_failed=True,
                vector_store_error_type=error_type,
                fallback_reason=fallback_reason,
            ),
        },
    )
    result = _empty_retrieval_result(
        query=query,
        top_k=top_k,
        request_id=request_id,
        latency_ms=latency_ms,
        retrieval_id=retrieval_event.id,
        document_id=document_id,
        session_id=session_id,
        session_scope=session_scope,
        strategy="rag_core_api",
        fallback_reason=fallback_reason,
        rag_mode=rag_mode,
        retrieval_scope=retrieval_scope,
    )
    result.update(
        build_retrieval_metadata_contract(
            rag_mode=rag_mode,
            retrieval_scope=retrieval_scope,
            selected_document_id=document_id,
            session_id=session_id,
            vector_store_attempted=False,
            vector_store_failed=True,
            vector_store_error_type=error_type,
            fallback_reason=fallback_reason,
        )
    )
    return result


def _safe_int_setting(name: str, default: int) -> int:
    value = getattr(settings, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dynamic_context_top_k(query: str, base_top_k: int) -> tuple[int, bool]:
    """Expand standard retrieval depth for count/list/summarize queries only."""
    if not detect_context_expansion_intent(query):
        return base_top_k, False
    max_context_chunks = max(base_top_k, _safe_int_setting("retrieval_max_context_chunks", 10))
    expanded = min(max_context_chunks, max(base_top_k, base_top_k * 2))
    return expanded, expanded > base_top_k


def _result_sort_key(item: dict) -> tuple[int, int, int]:
    return (
        int(item.get("document_id") or 0),
        int(item.get("chunk_index") or 0),
        int(item.get("chunk_id") or 0),
    )


def _apply_adjacent_chunk_expansion(
    db: Session,
    owner_username: str,
    results: list[dict],
    *,
    document_id: int | None,
    session_id: int | None,
    session_scope: str,
    max_chunks: int,
) -> tuple[list[dict], int]:
    """Add adjacent chunks for standard vector/hybrid retrieval within a safe cap."""
    if not results or max_chunks <= len(results):
        return results, 0

    anchor_ids = [int(result["chunk_id"]) for result in results if result.get("chunk_id") is not None and float(result.get("score") or 0.0) >= settings.retrieval_low_confidence_score]
    if not anchor_ids:
        return results, 0

    adjacent_rows = crud_knowledge.get_adjacent_chunks(
        db,
        owner_username,
        anchor_ids,
        window=1,
        document_id=document_id,
        session_id=session_id,
        session_scope=session_scope,
        indexed_only=True,
    )
    if not adjacent_rows:
        return results, 0

    by_id = {int(result["chunk_id"]): dict(result) for result in results if result.get("chunk_id") is not None}
    anchor_section_by_id = {
        int(result["chunk_id"]): result.get("section_key")
        for result in results
        if result.get("chunk_id") is not None and result.get("section_key")
    }
    anchor_doc_sections = {
        int(result.get("document_id") or 0): result.get("section_key")
        for result in results
        if result.get("section_key")
    }

    for chunk, document in adjacent_rows:
        chunk_id = int(chunk.id)
        if chunk_id in by_id:
            continue
        chunk_meta = ensure_json_mapping(chunk.metadata_json)
        expected_section = anchor_doc_sections.get(int(document.id))
        if expected_section and chunk_meta.get("section_key") != expected_section:
            continue
        by_id[chunk_id] = _build_result_from_chunk(chunk, document, score=0.0)

    expanded = sorted(by_id.values(), key=_result_sort_key)[:max_chunks]
    return expanded, max(0, len(expanded) - len(results))


# ---------------------------------------------------------------------------
# Main retrieval orchestrator
# ---------------------------------------------------------------------------

def search_knowledge(
    db: Session,
    owner_username: str,
    query: str,
    *,
    top_k: int | None = None,
    document_id: int | None = None,
    session_id: int | None = None,
    request_id: str | None = None,
) -> dict:
    normalized_owner = (owner_username or "").strip()
    normalized_query = " ".join((query or "").split())
    if not normalized_owner:
        raise ValueError("owner_username is required.")
    if not normalized_query:
        raise ValueError("query is required.")

    requested_top_k = top_k or settings.retrieval_top_k
    effective_top_k = requested_top_k
    started_at = time.perf_counter()
    retrieval_scope, rag_mode, session_scope = _resolve_retrieval_scope(document_id, session_id)

    if retrieval_scope == "session" and not crud_knowledge.has_indexed_documents_for_session(db, normalized_owner, session_id):
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        retrieval_event = crud_knowledge.create_retrieval_event(
            db,
            username=normalized_owner,
            query_text=normalized_query,
            top_k=effective_top_k,
            session_id=session_id,
            request_id=request_id,
            latency_ms=latency_ms,
            metadata_json={
                **build_retrieval_metadata_contract(
                    rag_mode="indexing_pending",
                    retrieval_scope="session",
                    session_id=session_id,
                    fallback_reason="no_indexed_session_documents",
                ),
                "document_id": document_id,
                "strategy": "indexing_pending",
                "original_query": normalized_query,
                "rewritten_query": normalized_query,
                "query_expansions": [],
                "returned": 0,
                "candidate_count": 0,
                "matched_count": 0,
                "reranked_count": 0,
                "fallback_used": True,
                "evidence_strength": "none",
                "session_scope": session_scope,
            },
        )
        return _empty_retrieval_result(
            query=normalized_query,
            top_k=effective_top_k,
            request_id=request_id,
            latency_ms=latency_ms,
            retrieval_id=retrieval_event.id,
            document_id=document_id,
            session_id=session_id,
            session_scope=session_scope,
            strategy="indexing_pending",
            fallback_reason="no_indexed_session_documents",
            rag_mode="indexing_pending",
            retrieval_scope="session",
        )

    available_sections = crud_knowledge.list_available_sections(
        db,
        normalized_owner,
        document_id=document_id,
        session_id=session_id,
        session_scope=session_scope,
        indexed_only=True,
    )
    section_match = match_section(normalized_query, available_sections)
    if section_match.section_key and section_match.confidence >= SECTION_CONFIDENCE_THRESHOLD:
        section_rows = crud_knowledge.find_chunks_by_section_key(
            db,
            normalized_owner,
            section_match.section_key,
            document_id=document_id,
            session_id=session_id,
            session_scope=session_scope,
            indexed_only=True,
        )
        if section_rows:
            results = [
                _build_result_from_chunk(chunk, document, score=section_match.confidence)
                for chunk, document in section_rows
            ][:effective_top_k]
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            retrieval_event = crud_knowledge.create_retrieval_event(
                db,
                username=normalized_owner,
                query_text=normalized_query,
                top_k=effective_top_k,
                session_id=session_id,
                request_id=request_id,
                latency_ms=latency_ms,
                metadata_json={
                    "document_id": document_id,
                    "strategy": "section_lookup",
                    "original_query": normalized_query,
                    "rewritten_query": normalized_query,
                    "query_expansions": [],
                    "returned": len(results),
                    "candidate_count": len(section_rows),
                    "matched_count": len(section_rows),
                    "reranked_count": len(results),
                    "fallback_used": False,
                    "fallback_reason": None,
                    "evidence_strength": "grounded",
                    "session_scope": session_scope,
                    **build_retrieval_metadata_contract(
                        rag_mode="section_rag",
                        retrieval_scope=retrieval_scope,
                        selected_document_id=document_id,
                        session_id=session_id,
                        section_key=section_match.section_key,
                        section_confidence=section_match.confidence,
                        vector_store_attempted=False,
                        vector_store_failed=False,
                    ),
                },
            )
            return {
                "query": normalized_query,
                "top_k": effective_top_k,
                "returned": len(results),
                "retrieval_id": retrieval_event.id,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "candidate_count": len(section_rows),
                "document_id": document_id,
                "strategy": "section_lookup",
                "original_query": normalized_query,
                "rewritten_query": normalized_query,
                "query_expansions": [],
                "reranked_count": len(results),
                "fallback_used": False,
                "fallback_reason": None,
                "evidence_strength": "grounded",
                "session_scope": session_scope,
                **build_retrieval_metadata_contract(
                    rag_mode="section_rag",
                    retrieval_scope=retrieval_scope,
                    selected_document_id=document_id,
                    session_id=session_id,
                    section_key=section_match.section_key,
                    section_confidence=section_match.confidence,
                    vector_store_attempted=False,
                    vector_store_failed=False,
                ),
                "results": results,
            }

    standard_top_k, dynamic_context_expanded = _dynamic_context_top_k(normalized_query, requested_top_k)
    effective_top_k = standard_top_k

    rewritten_query, query_expansions = _expand_query(normalized_query)
    rag_core_api_mode = is_rag_core_api_mode()
    rag_core_client = get_rag_core_client() if rag_core_api_mode else None

    if rag_core_api_mode:
        _embed_provider = None
        query_embedding = []
        _embed_meta = _embedding_meta_namespace()
    else:
        # P01-T07: use provider factory so query and document embeddings share the same space
        _embed_provider = get_embedding_provider()
        _query_embed_result = _embed_provider.embed_query(rewritten_query)
        query_embedding = _query_embed_result.vector
        _embed_meta = _query_embed_result.meta

    semantic_scores_by_chunk_id: dict[int, float] = {}
    candidates = []
    vector_store_attempted = False
    vector_store_failed = False
    vector_store_error_type = None

    if vector_store.is_external_vector_store_enabled():
        vector_store_attempted = True
        try:
            if rag_core_api_mode and rag_core_client is not None:
                vector_response = rag_core_client.vector_search(
                    owner_username=normalized_owner,
                    query=normalized_query,
                    rewritten_query=rewritten_query,
                    top_k=max(effective_top_k, settings.retrieval_max_rerank_candidates),
                    document_id=document_id,
                    session_id=session_id,
                    session_scope=session_scope,
                    request_id=request_id,
                )
                rewritten_query = vector_response.get("rewritten_query") or rewritten_query
                query_expansions = vector_response.get("query_expansions") or query_expansions
                _embed_meta = _embedding_meta_namespace(vector_response.get("embedding_meta") or {})
                vector_store_failed = bool(vector_response.get("vector_store_failed"))
                vector_store_error_type = vector_response.get("vector_store_error_type")
                vector_hits = vector_response.get("vector_hits") or []
            else:
                vector_hits = vector_store.search_similar_chunks(
                    normalized_owner,
                    query_embedding,
                    top_k=max(effective_top_k, settings.retrieval_max_rerank_candidates),
                    document_id=document_id,
                    session_id=session_id,
                    session_scope=session_scope,
                )
            semantic_scores_by_chunk_id = {
                int(hit["chunk_id"]): round(max(0.0, float(hit.get("score") or 0.0)), 6)
                for hit in vector_hits
            }
            chunk_order = {
                int(hit["chunk_id"]): index
                for index, hit in enumerate(vector_hits)
            }
            candidates = crud_knowledge.list_searchable_chunks_by_ids(
                db,
                normalized_owner,
                list(chunk_order.keys()),
                document_id=document_id,
                session_id=session_id,
                session_scope=session_scope,
                indexed_only=True,
            )
            candidates.sort(key=lambda row: chunk_order.get(int(row[0].id), len(chunk_order)))
        except Exception as exc:
            vector_store_failed = True
            vector_store_error_type = type(exc).__name__
            logger.warning(
                "Qdrant retrieval failed, falling back to database search: "
                "error_type=%s collection=%s top_k=%s",
                type(exc).__name__,
                settings.vector_store_collection,
                max(effective_top_k, settings.retrieval_max_rerank_candidates),
            )
            semantic_scores_by_chunk_id = {}

    if not candidates:
        candidates = crud_knowledge.list_searchable_chunks(
            db,
            normalized_owner,
            document_id=document_id,
            session_id=session_id,
            session_scope=session_scope,
            indexed_only=True,
        )

    _mixed_space_skip_count = 0
    matched_count = 0
    reranked_count = 0

    if rag_core_api_mode and rag_core_client is not None:
        try:
            rank_response = rag_core_client.rank_retrieval(
                query=normalized_query,
                top_k=effective_top_k,
                candidates=_serialize_retrieval_candidates(candidates),
                rewritten_query=rewritten_query,
                query_expansions=query_expansions,
                semantic_scores_by_chunk_id=semantic_scores_by_chunk_id,
                embedding_meta={
                    "provider": _embed_meta.provider,
                    "model": _embed_meta.model,
                    "dimensions": _embed_meta.dimensions,
                    "version": _embed_meta.version,
                    "extra": _embed_meta.extra,
                },
                request_id=request_id,
            )
        except RagCoreClientError as exc:
            logger.warning(
                "rag-core API retrieval ranking failed: error_type=%s",
                exc.error_type,
            )
            return _rag_core_api_unavailable_result(
                db,
                owner_username=normalized_owner,
                query=normalized_query,
                top_k=effective_top_k,
                request_id=request_id,
                started_at=started_at,
                document_id=document_id,
                session_id=session_id,
                session_scope=session_scope,
                retrieval_scope=retrieval_scope,
                rag_mode=rag_mode,
                error_type=exc.error_type,
            )
        results = rank_response.get("results") or []
        matched_count = int(rank_response.get("matched_count") or 0)
        reranked_count = int(rank_response.get("reranked_count") or 0)
        rewritten_query = rank_response.get("rewritten_query") or rewritten_query
        query_expansions = rank_response.get("query_expansions") or query_expansions
        _mixed_space_skip_count = int(rank_response.get("mixed_space_skip_count") or 0)
        _embed_meta = _embedding_meta_namespace(rank_response.get("embedding_meta") or {})
    else:
        scored_results: list[dict] = []
        # P05-T01: capture current provider identity for mixed-space check
        _query_identity = (_embed_meta.provider, _embed_meta.model)
        chunk_meta_by_id: dict[int, dict] = {}
        chunk_embeddings_by_id: dict[int, list[float]] = {}
        missing_embedding_inputs: list[tuple[int, str]] = []

        if not semantic_scores_by_chunk_id:
            for chunk, _document in candidates:
                chunk_id = int(chunk.id)
                chunk_meta = ensure_json_mapping(chunk.metadata_json)
                chunk_meta_by_id[chunk_id] = chunk_meta
                if not _is_embedding_compatible(chunk_meta, _query_identity[0], _query_identity[1]):
                    _mixed_space_skip_count += 1
                    continue

                stored_embedding = _extract_stored_embedding(chunk_meta)
                if stored_embedding is not None:
                    chunk_embeddings_by_id[chunk_id] = stored_embedding
                else:
                    missing_embedding_inputs.append((chunk_id, chunk.content))

            if missing_embedding_inputs and _embed_provider is not None:
                embed_result = _embed_provider.embed_texts(
                    [content for _chunk_id, content in missing_embedding_inputs]
                )
                for (chunk_id, _content), vector in zip(missing_embedding_inputs, embed_result.vectors):
                    chunk_embeddings_by_id[chunk_id] = vector

        for chunk, document in candidates:
            # P06 (FINDING-R15): cache parsed chunk_meta to avoid double ensure_json_mapping
            chunk_id = int(chunk.id)
            chunk_meta = chunk_meta_by_id.get(chunk_id) or ensure_json_mapping(chunk.metadata_json)
            if semantic_scores_by_chunk_id:
                semantic_score = semantic_scores_by_chunk_id.get(chunk_id, 0.0)
            elif _is_embedding_compatible(chunk_meta, _query_identity[0], _query_identity[1]):
                chunk_embedding = chunk_embeddings_by_id.get(chunk_id)
                if chunk_embedding is not None:
                    semantic_score = round(max(0.0, _cosine_similarity(query_embedding, chunk_embedding)), 6)
                else:
                    semantic_score = 0.0
            else:
                semantic_score = 0.0
            lexical_score = _lexical_overlap_score(rewritten_query, chunk.content)
            score = _hybrid_score(semantic_score, lexical_score)
            if score < settings.retrieval_min_score and lexical_score < settings.retrieval_min_lexical_score:
                continue
            scored_results.append(
                {
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "title": document.title,
                    "source_type": document.source_type,
                    "source_uri": document.source_uri,
                    "score": score,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "token_count": int(chunk.token_count or 0) if getattr(chunk, "token_count", None) is not None else None,
                    "snippet": _build_snippet(chunk.content),
                    "content": chunk.content,
                    "vector_id": chunk.vector_id,
                    "embedding_model": chunk.embedding_model,
                    "embedding_provider": chunk_meta.get("embedding_provider", _embed_meta.provider),
                    **_extract_chunk_metadata_fields(chunk_meta),
                }
            )

        scored_results.sort(key=lambda item: (-item["score"], item["document_id"], item["chunk_index"]))
        deduped_results = _dedupe_scored_results(scored_results)
        reranked_results = _rerank_results(rewritten_query, deduped_results)
        results = reranked_results[:effective_top_k]
        matched_count = len(deduped_results)
        reranked_count = len(reranked_results)
    max_context_chunks = max(effective_top_k, _safe_int_setting("retrieval_max_context_chunks", 10))
    results, adjacent_expanded_count = _apply_adjacent_chunk_expansion(
        db,
        normalized_owner,
        results,
        document_id=document_id,
        session_id=session_id,
        session_scope=session_scope,
        max_chunks=max_context_chunks,
    )

    fallback_used = False
    fallback_candidates_source = candidates
    if not results and document_id is not None:
        if not fallback_candidates_source:
            fallback_candidates_source = crud_knowledge.list_searchable_chunks(
                db,
                normalized_owner,
                document_id=document_id,
                session_id=session_id,
                session_scope=session_scope,
                indexed_only=True,
            )

    if not results and document_id is not None and fallback_candidates_source:
        fallback_candidates: list[dict] = []
        for chunk, document in fallback_candidates_source:
            chunk_meta = ensure_json_mapping(chunk.metadata_json)
            fallback_candidates.append(
                {
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "title": document.title,
                    "source_type": document.source_type,
                    "source_uri": document.source_uri,
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "rerank_score": 0.0,
                    "token_estimate": _estimate_token_count(chunk.content, int(chunk.token_count or 0) if getattr(chunk, "token_count", None) is not None else None),
                    "snippet": _build_snippet(chunk.content),
                    "content": chunk.content,
                    "vector_id": chunk.vector_id,
                    "embedding_model": chunk.embedding_model,
                    "embedding_provider": chunk_meta.get("embedding_provider", _embed_meta.provider),
                    **_extract_chunk_metadata_fields(chunk_meta),
                }
            )
        fallback_candidates.sort(key=lambda item: (item["document_id"], item["chunk_index"]))
        results = fallback_candidates[:effective_top_k]
        fallback_used = bool(results)

    evidence_strength = _classify_evidence_strength(results, fallback_used=fallback_used)
    strategy = "hybrid_rerank"
    fallback_reason = None
    if fallback_used:
        fallback_reason = "document_scope_seed"
    elif vector_store_failed:
        fallback_reason = "vector_store_failure"
    elif available_sections and section_match.intent.is_section_query:
        fallback_reason = "low_section_confidence"
    elif not results:
        fallback_reason = "no_relevant_match"
    elif evidence_strength == "weak":
        fallback_reason = "low_confidence_match"

    latency_ms = int((time.perf_counter() - started_at) * 1000)

    retrieval_event = crud_knowledge.create_retrieval_event(
        db,
        username=normalized_owner,
        query_text=normalized_query,
        top_k=effective_top_k,
        session_id=session_id,
        request_id=request_id,
        latency_ms=latency_ms,
        metadata_json={
            "document_id": document_id,
            "strategy": strategy,
            "original_query": normalized_query,
            "rewritten_query": rewritten_query,
            "query_expansions": query_expansions,
            "returned": len(results),
            "candidate_count": len(candidates),
            "matched_count": matched_count,
            "reranked_count": reranked_count,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "evidence_strength": evidence_strength,
            "dynamic_context_expanded": dynamic_context_expanded,
            "requested_top_k": requested_top_k,
            "effective_top_k": effective_top_k,
            "adjacent_expansion_applied": adjacent_expanded_count > 0,
            "adjacent_expanded_count": adjacent_expanded_count,
            **build_retrieval_metadata_contract(
                rag_mode=rag_mode,
                retrieval_scope=retrieval_scope,
                selected_document_id=document_id,
                session_id=session_id,
                section_key=section_match.section_key if section_match.intent.is_section_query else None,
                section_confidence=section_match.confidence if section_match.intent.is_section_query else None,
                vector_store_attempted=vector_store_attempted,
                vector_store_failed=vector_store_failed,
                vector_store_error_type=vector_store_error_type,
                fallback_reason=fallback_reason,
            ),
            # P01-T07: provider metadata from factory (additive, no schema change)
            "embedding_provider": _embed_meta.provider,
            "embedding_model": _embed_meta.model,
            "embedding_dimensions": _embed_meta.dimensions,
            "embedding_version": _embed_meta.version,
            "vector_store_provider": settings.vector_store_provider,
            "vector_store_collection": settings.vector_store_collection,
            "session_scope": session_scope,
            # P05-T01: api_type from provider metadata (extra dict from generic api provider)
            "api_type": _embed_meta.extra.get("api_type", ""),
            # P05-T01: mixed-space safeguard tracking
            "mixed_space_skip_count": _mixed_space_skip_count,
        },
    )

    return {
        "query": normalized_query,
        "top_k": effective_top_k,
        "returned": len(results),
        "retrieval_id": retrieval_event.id,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "candidate_count": len(candidates),
        "document_id": document_id,
        "strategy": strategy,
        "original_query": normalized_query,
        "rewritten_query": rewritten_query,
        "query_expansions": query_expansions,
        "reranked_count": reranked_count,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "evidence_strength": evidence_strength,
        "dynamic_context_expanded": dynamic_context_expanded,
        "requested_top_k": requested_top_k,
        "effective_top_k": effective_top_k,
        "adjacent_expansion_applied": adjacent_expanded_count > 0,
        "adjacent_expanded_count": adjacent_expanded_count,
        "session_scope": session_scope,
        **build_retrieval_metadata_contract(
            rag_mode=rag_mode,
            retrieval_scope=retrieval_scope,
            selected_document_id=document_id,
            session_id=session_id,
            section_key=section_match.section_key if section_match.intent.is_section_query else None,
            section_confidence=section_match.confidence if section_match.intent.is_section_query else None,
            vector_store_attempted=vector_store_attempted,
            vector_store_failed=vector_store_failed,
            vector_store_error_type=vector_store_error_type,
            fallback_reason=fallback_reason,
        ),
        "results": results,
    }
