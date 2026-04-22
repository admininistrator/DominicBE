"""Knowledge retrieval service for Phase 2 searchable indexing."""
from __future__ import annotations

import unicodedata
import time
from math import sqrt
import re

from sqlalchemy.orm import Session

from app.core.config import settings
from app.crud import crud_knowledge
from app.services.knowledge_service import compute_text_embedding


QUERY_EXPANSION_RULES: dict[str, list[str]] = {
    "hoan tien": ["refund", "refund policy", "money back"],
    "chinh sach": ["policy"],
    "xu ly": ["review", "process", "processing"],
    "bao lau": ["how long", "duration", "timeline", "days"],
    "mat khau": ["password", "credentials"],
    "dang nhap": ["login", "sign in", "authenticate"],
    "tai lieu": ["document", "knowledge base"],
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_for_search(text: str) -> str:
    lowered = _strip_accents(text).lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", _normalize_for_search(text), flags=re.UNICODE) if token}


def _expand_query(query: str) -> tuple[str, list[str]]:
    normalized = _normalize_for_search(query)
    expansions: list[str] = []
    if settings.retrieval_enable_query_expansion:
        for phrase, candidates in QUERY_EXPANSION_RULES.items():
            if phrase in normalized:
                for candidate in candidates:
                    if candidate not in expansions:
                        expansions.append(candidate)

    if not expansions:
        return " ".join((query or "").split()), []

    rewritten_query = " ".join(part for part in [" ".join((query or "").split()), *expansions] if part).strip()
    return rewritten_query, expansions


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _extract_embedding(metadata_json: dict | None, fallback_text: str) -> list[float]:
    embedding = (metadata_json or {}).get("embedding")
    if isinstance(embedding, list) and embedding:
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError):
            pass
    return compute_text_embedding(fallback_text)


def _build_snippet(content: str, *, max_chars: int = 220) -> str:
    normalized = " ".join((content or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _estimate_token_count(text: str, explicit_count: int | None = None) -> int:
    if explicit_count and explicit_count > 0:
        return int(explicit_count)
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _lexical_overlap_score(query_text: str, content: str) -> float:
    query_tokens = _tokenize(query_text)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query_tokens)
    density = len(overlap) / sqrt(len(query_tokens) * len(content_tokens))
    return round(min(1.0, (coverage * 0.75) + (density * 0.25)), 6)


def _hybrid_score(semantic_score: float, lexical_score: float) -> float:
    total_weight = settings.retrieval_hybrid_semantic_weight + settings.retrieval_hybrid_lexical_weight
    semantic_weight = settings.retrieval_hybrid_semantic_weight / total_weight
    lexical_weight = settings.retrieval_hybrid_lexical_weight / total_weight
    return round(
        min(1.0, (semantic_score * semantic_weight) + (lexical_score * lexical_weight)),
        6,
    )


def _classify_evidence_strength(results: list[dict], *, fallback_used: bool) -> str:
    if not results:
        return "none"
    if fallback_used:
        return "fallback"
    top_score = float(results[0].get("score") or 0.0)
    if top_score >= settings.retrieval_low_confidence_score:
        return "grounded"
    return "weak"


def _rerank_results(query_text: str, results: list[dict]) -> list[dict]:
    reranked: list[dict] = []
    for item in results[: settings.retrieval_max_rerank_candidates]:
        title_score = _lexical_overlap_score(query_text, item.get("title") or "")
        chunk_index = int(item.get("chunk_index") or 0)
        position_score = max(0.0, 1.0 - (chunk_index * 0.05))
        rerank_score = round(
            min(
                1.0,
                float(item.get("score") or 0.0)
                + (title_score * settings.retrieval_rerank_title_weight)
                + (position_score * settings.retrieval_rerank_position_weight),
            ),
            6,
        )
        reranked.append(
            {
                **item,
                "rerank_score": rerank_score,
                "token_estimate": _estimate_token_count(item.get("content") or "", item.get("token_count")),
            }
        )

    reranked.sort(
        key=lambda item: (
            -float(item.get("rerank_score") or 0.0),
            -float(item.get("score") or 0.0),
            item.get("document_id") or 0,
            item.get("chunk_index") or 0,
        )
    )
    return reranked


def _dedupe_scored_results(results: list[dict]) -> list[dict]:
    seen: set[tuple[int, str]] = set()
    deduped: list[dict] = []

    for item in results:
        key = (item["document_id"], _normalize_for_dedupe(item.get("content") or item.get("snippet") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


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
    query_embedding = compute_text_embedding(rewritten_query)

    candidates = crud_knowledge.list_searchable_chunks(
        db,
        normalized_owner,
        document_id=document_id,
        indexed_only=True,
    )

    scored_results: list[dict] = []
    for chunk, document in candidates:
        chunk_embedding = _extract_embedding(chunk.metadata_json, chunk.content)
        semantic_score = round(max(0.0, _cosine_similarity(query_embedding, chunk_embedding)), 6)
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
            }
        )

    scored_results.sort(key=lambda item: (-item["score"], item["document_id"], item["chunk_index"]))
    deduped_results = _dedupe_scored_results(scored_results)
    reranked_results = _rerank_results(rewritten_query, deduped_results)
    results = reranked_results[:effective_top_k]

    fallback_used = False
    if not results and document_id is not None and candidates:
        fallback_candidates: list[dict] = []
        for chunk, document in candidates:
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
            "embedding_provider": settings.embedding_provider,
            "vector_store_provider": settings.vector_store_provider,
        },
    )

    return {
        "query": normalized_query,
        "top_k": effective_top_k,
        "returned": len(results),
        "retrieval_id": retrieval_event.id,
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
        "results": results,
    }

