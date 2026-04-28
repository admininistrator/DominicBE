import logging
import time
from urllib.parse import urlparse

import httpx

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


class TavilySearchError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


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

    search_url = settings.tavily_base_url.rstrip("/") + "/search"
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "topic": settings.web_search_topic,
        "search_depth": settings.tavily_search_depth,
        "max_results": max_results or settings.web_search_max_results,
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
    results = [
        _normalize_tavily_result(row, index)
        for index, row in enumerate(data.get("results") or [], start=1)
        if isinstance(row, dict)
    ]

    return {
        "used": True,
        "query": (data.get("query") or query or "").strip(),
        "latency_ms": latency_ms,
        "results": results,
    }