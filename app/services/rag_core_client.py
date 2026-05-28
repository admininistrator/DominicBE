"""HTTP client for the internal rag-core API service."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings


class RagCoreClientError(RuntimeError):
    """Raised when the rag-core API cannot satisfy a request."""

    def __init__(self, message: str, *, status_code: int | None = None, error_type: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type or "RagCoreClientError"


def is_rag_core_api_mode() -> bool:
    return (settings.rag_core_mode or "library").strip().lower() == "api"


class RagCoreClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout_seconds
        if not self._base_url:
            raise RagCoreClientError("RAG_CORE_BASE_URL is not configured.", error_type="configuration_error")
        if not self._api_key:
            raise RagCoreClientError("RAG_CORE_API_KEY is required when RAG_CORE_MODE=api.", error_type="configuration_error")

    def health(self) -> dict:
        return self._request("GET", "/health", auth=False)

    def prepare_indexing(
        self,
        *,
        document_id: int,
        checksum: str,
        chunks: list[dict],
        store_embeddings_in_metadata: bool | None,
        index_provider: str,
        request_id: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/indexing/prepare",
            json={
                "document_id": document_id,
                "checksum": checksum,
                "chunks": chunks,
                "store_embeddings_in_metadata": store_embeddings_in_metadata,
                "index_provider": index_provider,
            },
            request_id=request_id,
        )

    def vector_upsert(
        self,
        *,
        document: dict,
        chunk_rows: list[dict],
        prepared_chunks: list[dict],
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/vector/upsert",
            json={
                "document": document,
                "chunk_rows": chunk_rows,
                "prepared_chunks": prepared_chunks,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
            },
            request_id=request_id,
        )

    def vector_delete(self, *, owner_username: str, document_id: int, request_id: str | None = None) -> dict:
        return self._request(
            "POST",
            "/v1/vector/delete",
            json={"owner_username": owner_username, "document_id": document_id},
            request_id=request_id,
        )

    def vector_search(
        self,
        *,
        owner_username: str,
        top_k: int,
        query: str | None = None,
        rewritten_query: str | None = None,
        query_vector: list[float] | None = None,
        document_id: int | None = None,
        session_id: int | None = None,
        session_scope: str = "all",
        request_id: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/vector/search",
            json={
                "owner_username": owner_username,
                "top_k": top_k,
                "query": query,
                "rewritten_query": rewritten_query,
                "query_vector": query_vector,
                "document_id": document_id,
                "session_id": session_id,
                "session_scope": session_scope,
            },
            request_id=request_id,
        )

    def rank_retrieval(
        self,
        *,
        query: str,
        top_k: int,
        candidates: list[dict],
        rewritten_query: str | None,
        query_expansions: list[str],
        semantic_scores_by_chunk_id: dict[int, float],
        embedding_meta: dict | None,
        request_id: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/retrieval/rank",
            json={
                "query": query,
                "top_k": top_k,
                "candidates": candidates,
                "rewritten_query": rewritten_query,
                "query_expansions": query_expansions,
                "semantic_scores_by_chunk_id": {
                    str(chunk_id): score for chunk_id, score in semantic_scores_by_chunk_id.items()
                },
                "embedding_meta": embedding_meta,
            },
            request_id=request_id,
        )

    def pack_context(
        self,
        *,
        results: list[dict],
        max_context_chunks: int,
        max_context_tokens: int,
        request_id: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/v1/context/pack",
            json={
                "results": results,
                "max_context_chunks": max_context_chunks,
                "max_context_tokens": max_context_tokens,
            },
            request_id=request_id,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        request_id: str | None = None,
        auth: bool = True,
    ) -> dict:
        headers: dict[str, str] = {}
        if auth:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
                response = client.request(method, path, json=json, headers=headers)
        except httpx.TimeoutException as exc:
            raise RagCoreClientError("rag-core API request timed out.", error_type=type(exc).__name__) from exc
        except httpx.RequestError as exc:
            raise RagCoreClientError("rag-core API request failed.", error_type=type(exc).__name__) from exc

        if response.status_code >= 400:
            error_type = "HTTPError"
            try:
                body = response.json()
                detail = body.get("detail") if isinstance(body, dict) else None
                if isinstance(detail, dict):
                    error_type = str(detail.get("error_type") or error_type)
            except ValueError:
                pass
            raise RagCoreClientError(
                f"rag-core API returned HTTP {response.status_code}.",
                status_code=response.status_code,
                error_type=error_type,
            )

        try:
            parsed = response.json()
        except ValueError as exc:
            raise RagCoreClientError("rag-core API returned invalid JSON.", error_type="invalid_json") from exc
        if not isinstance(parsed, dict):
            raise RagCoreClientError("rag-core API returned a non-object payload.", error_type="invalid_payload")
        return parsed


def get_rag_core_client() -> RagCoreClient:
    return RagCoreClient(
        base_url=settings.rag_core_base_url,
        api_key=settings.rag_core_api_key,
        timeout_seconds=settings.rag_core_timeout_seconds,
    )

