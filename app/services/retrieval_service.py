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
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.json_utils import ensure_json_mapping
from app.crud import crud_knowledge
from app.services import vector_store
from app.services.embeddings.factory import get_embedding_provider

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

logger = get_logger(__name__)


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

    effective_top_k = top_k or settings.retrieval_top_k
    started_at = time.perf_counter()
    rewritten_query, query_expansions = _expand_query(normalized_query)

    # P01-T07: use provider factory so query and document embeddings share the same space
    _embed_provider = get_embedding_provider()
    _query_embed_result = _embed_provider.embed_query(rewritten_query)
    query_embedding = _query_embed_result.vector
    _embed_meta = _query_embed_result.meta

    session_scope = "all"
    if document_id is None and session_id is not None:
        session_scope = (
            "session"
            if crud_knowledge.has_indexed_documents_for_session(db, normalized_owner, session_id)
            else "global"
        )

    semantic_scores_by_chunk_id: dict[int, float] = {}
    candidates = []

    if vector_store.is_external_vector_store_enabled():
        try:
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
            logger.warning("Qdrant retrieval failed, falling back to database search: %s", exc)
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

        if missing_embedding_inputs:
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
            }
        )

    scored_results.sort(key=lambda item: (-item["score"], item["document_id"], item["chunk_index"]))
    deduped_results = _dedupe_scored_results(scored_results)
    reranked_results = _rerank_results(rewritten_query, deduped_results)
    results = reranked_results[:effective_top_k]

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
            "matched_count": len(deduped_results),
            "reranked_count": len(reranked_results),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "evidence_strength": evidence_strength,
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
        "reranked_count": len(reranked_results),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "evidence_strength": evidence_strength,
        "session_scope": session_scope,
        "results": results,
    }
