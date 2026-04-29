import logging
import re
import time
import json
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.services import llm_provider
from app.services.llm_provider import LLMError

logger = logging.getLogger("uvicorn.error")

COMMON_FILLER_TERMS = {
    "la",
    "là",
    "gi",
    "gì",
    "cua",
    "của",
    "va",
    "và",
    "voi",
    "với",
    "ket",
    "kết",
    "hop",
    "hợp",
    "su",
    "sử",
    "dung",
    "dụng",
    "nhung",
    "những",
    "nao",
    "nào",
    "the",
    "thế",
    "what",
    "is",
    "and",
    "with",
    "use",
    "using",
}


class TavilySearchError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _normalize_search_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _extract_branded_terms(query: str) -> list[str]:
    matches = re.findall(r"\b(?:[A-Z][A-Za-z0-9.-]{2,}|[A-Za-z]+[A-Z][A-Za-z0-9.-]*|[0-9][A-Za-z0-9.-]+)\b", query)
    seen: set[str] = set()
    results: list[str] = []
    for match in matches:
        normalized_key = match.casefold()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        results.append(match)
    return results


def _dedupe_search_queries(candidates: list[str], *, max_queries: int) -> list[str]:
    deduped_queries: list[str] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        normalized_candidate = _normalize_search_text(candidate)
        if not normalized_candidate:
            continue
        key = normalized_candidate.casefold()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        deduped_queries.append(normalized_candidate)
        if len(deduped_queries) >= max_queries:
            break
    return deduped_queries


def _build_heuristic_search_queries(query: str) -> list[str]:
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return []

    branded_terms = _extract_branded_terms(normalized_query)
    queries: list[str] = [normalized_query]

    if len(branded_terms) > 1:
        queries.append(" ".join(branded_terms))

    for term in branded_terms:
        queries.append(term)

    phrase_candidates = re.split(r"[,;?.!]|\b(?:và|va|and|with|cùng|cung|kết hợp|ket hop|using)\b", normalized_query, flags=re.IGNORECASE)
    for candidate in phrase_candidates:
        cleaned = _normalize_search_text(candidate)
        if len(cleaned) < 6:
            continue
        words = cleaned.split()
        if all(word.casefold() in COMMON_FILLER_TERMS for word in words):
            continue
        queries.append(cleaned)

    return _dedupe_search_queries(
        queries,
        max_queries=settings.web_search_query_planner_max_queries,
    )


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _plan_search_queries_with_model(query: str) -> list[str]:
    if not settings.web_search_query_planner_enabled:
        return []

    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return []

    branded_terms = _extract_branded_terms(normalized_query)
    planner_system = (
        "You rewrite a user question into 2 to 4 high-quality web search queries for Tavily. "
        "Preserve product names, organization names, model names, code names, URLs, acronyms, and technical brands exactly when they appear. "
        "Do not translate or normalize branded technical terms. "
        "Prefer short, search-engine-friendly queries. "
        "Return strict JSON only in the shape {\"queries\": [\"...\", \"...\"]}."
    )
    planner_user = (
        f"Original user question: {normalized_query}\n"
        f"Known branded terms that must be preserved exactly when relevant: {', '.join(branded_terms) if branded_terms else '(none)'}\n"
        f"Generate between 2 and {settings.web_search_query_planner_max_queries} search queries."
    )

    try:
        planner_result = llm_provider.complete(
            messages=[{"role": "user", "content": planner_user}],
            system=planner_system,
            max_tokens=220,
            model=settings.web_search_query_planner_model,
        )
    except LLMError as exc:
        logger.warning("Web query planner failed, falling back to heuristic queries: %s", exc.detail)
        return []
    except Exception as exc:
        logger.warning("Web query planner crashed, falling back to heuristic queries: %s", exc, exc_info=True)
        return []

    payload = _extract_json_object(planner_result.get("text") or "") or {}
    raw_queries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(raw_queries, list):
        return []

    cleaned_queries = [item for item in raw_queries if isinstance(item, str)]
    return _dedupe_search_queries(
        cleaned_queries,
        max_queries=settings.web_search_query_planner_max_queries,
    )


def build_search_queries(query: str) -> list[str]:
    heuristic_queries = _build_heuristic_search_queries(query)
    planned_queries = _plan_search_queries_with_model(query)

    combined = list(planned_queries)
    minimum_queries = min(2, settings.web_search_query_planner_max_queries)
    if len(combined) < minimum_queries:
        combined.extend(heuristic_queries)
    else:
        combined.extend(heuristic_queries[:1])

    final_queries = _dedupe_search_queries(
        combined,
        max_queries=settings.web_search_query_planner_max_queries,
    )
    return final_queries or heuristic_queries


def _request_tavily(query: str, *, max_results: int) -> tuple[dict, int]:
    search_url = settings.tavily_base_url.rstrip("/") + "/search"
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "topic": settings.web_search_topic,
        "search_depth": settings.tavily_search_depth,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    started_at = time.perf_counter()
    try:
        with httpx.Client(timeout=settings.tavily_timeout_seconds) as client:
            response = client.post(search_url, json=payload)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        raise TavilySearchError(504, "Tavily request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        detail = "Tavily search request failed."
        try:
            response_payload = exc.response.json()
            detail = response_payload.get("detail") or response_payload.get("error") or detail
        except Exception:
            detail = exc.response.text.strip() or detail
        raise TavilySearchError(exc.response.status_code, detail) from exc
    except Exception as exc:
        logger.warning("Tavily request failed: %s", exc, exc_info=True)
        raise TavilySearchError(502, "Failed to call Tavily web search.") from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return data, latency_ms


def _normalize_tavily_result(row: dict, index: int) -> dict:
    url = (row.get("url") or "").strip()
    parsed = urlparse(url) if url else None
    domain = (parsed.netloc or "").removeprefix("www.") if parsed else None
    title = (row.get("title") or url or f"Web result {index}").strip()
    snippet = (row.get("content") or row.get("snippet") or "").strip()
    raw_score = row.get("score")

    return {
        "title": title,
        "url": url or None,
        "domain": domain or None,
        "snippet": snippet,
        "score": float(raw_score) if raw_score is not None else None,
        "published_date": row.get("published_date"),
    }


def search_web(query: str, *, max_results: int | None = None) -> dict:
    if not settings.web_search_enabled:
        raise TavilySearchError(503, "Web search is disabled in backend configuration.")

    if not settings.tavily_api_key:
        raise TavilySearchError(503, "Tavily API key is not configured. Please set TAVILY_API_KEY in .env.")

    requested_results = max_results or settings.web_search_max_results
    planned_queries = build_search_queries(query)
    per_query_limit = requested_results if len(planned_queries) == 1 else min(3, requested_results)

    logger.info("Tavily planned queries=%s", planned_queries)

    total_latency_ms = 0
    raw_results: list[dict] = []
    seen_urls: set[str] = set()
    primary_query = _normalize_search_text(query)

    for planned_query in planned_queries:
        data, latency_ms = _request_tavily(planned_query, max_results=per_query_limit)
        total_latency_ms += latency_ms
        if not primary_query:
            primary_query = _normalize_search_text(data.get("query") or planned_query)

        for row in data.get("results") or []:
            if not isinstance(row, dict):
                continue
            url = _normalize_search_text(row.get("url"))
            dedupe_key = url.casefold() if url else _normalize_search_text(row.get("title")).casefold()
            if dedupe_key and dedupe_key in seen_urls:
                continue
            if dedupe_key:
                seen_urls.add(dedupe_key)
            raw_results.append(row)
            if len(raw_results) >= requested_results:
                break
        if len(raw_results) >= requested_results:
            break

    results = [
        _normalize_tavily_result(row, index)
        for index, row in enumerate(raw_results, start=1)
    ]

    return {
        "used": True,
        "query": primary_query,
        "queries": planned_queries,
        "latency_ms": total_latency_ms,
        "results": results,
    }