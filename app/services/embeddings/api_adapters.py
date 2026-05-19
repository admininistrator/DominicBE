"""Request/response format adapters for API-based embedding providers.

Each adapter encapsulates the format differences for a specific provider API.
The ``get_api_adapter()`` function selects the correct adapter based on
``EMBEDDING_API_TYPE`` config. Unknown types fall back to OpenAI format.

Adapter interface:
    - ``format_request(texts, model, dimensions) -> (endpoint_path, request_body)``
    - ``parse_response(data, expected_count) -> list[list[float]]``

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


class APIAdapter(Protocol):
    """Protocol for API format adapters."""

    def format_request(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[str, dict]:
        """Build the request body and endpoint path for an embedding API call.

        Args:
            texts: Ordered list of text strings to embed.
            model: Model name.
            dimensions: Requested embedding dimensions (None or 0 = use default).

        Returns:
            Tuple of ``(endpoint_path, request_body_dict)``.
        """
        ...

    def parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        """Extract ordered embedding vectors from a provider API response.

        Args:
            data: Parsed JSON response dict from the provider API.
            expected_count: Number of embeddings expected (len of input texts).

        Returns:
            List of float vectors in the same order as input texts.

        Raises:
            ValueError: If the response cannot be parsed correctly.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (default)
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """OpenAI-compatible embedding API adapter.

    Endpoint: ``POST /v1/embeddings``
    Request: ``{"input": [...], "model": "...", "dimensions": N}``
    Response: ``{"data": [{"embedding": [...], "index": N}], "model": "...",
               "usage": {...}}``
    """

    def format_request(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[str, dict]:
        body: dict = {"input": texts, "model": model}
        if dimensions and dimensions > 0:
            body["dimensions"] = dimensions
        return "/v1/embeddings", body

    def parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        raw_embeddings = data.get("data")
        if not isinstance(raw_embeddings, list):
            raise ValueError(
                "OpenAI response missing 'data' list"
            )
        # Sort by index to preserve input order
        sorted_by_index = sorted(raw_embeddings, key=lambda x: x.get("index", 0))
        vectors: list[list[float]] = []
        for item in sorted_by_index:
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise ValueError(
                    f"OpenAI response item missing 'embedding' list: {item}"
                )
            vectors.append([float(v) for v in vec])

        if len(vectors) != expected_count:
            raise ValueError(
                f"OpenAI returned {len(vectors)} embeddings but expected {expected_count}"
            )
        return vectors


# ---------------------------------------------------------------------------
# Cohere adapter
# ---------------------------------------------------------------------------

class CohereAdapter:
    """Cohere embedding API adapter.

    Endpoint: ``POST /v1/embed``
    Request: ``{"texts": [...], "model": "...", "input_type": "search_document",
               "embedding_types": ["float"]}``
    Response: ``{"embeddings": {"float": [[...]]} or {"embeddings": [[...]]}}``
    """

    def format_request(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[str, dict]:
        body: dict = {
            "texts": texts,
            "model": model,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }
        if dimensions and dimensions > 0:
            body["dimensions"] = dimensions
        return "/v1/embed", body

    def parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        raw = data.get("embeddings")
        if isinstance(raw, dict) and "float" in raw:
            # Newer Cohere format: {"embeddings": {"float": [[...], ...]}}
            vectors_list = raw["float"]
        elif isinstance(raw, list):
            # Older Cohere format: {"embeddings": [[...], ...]}
            vectors_list = raw
        else:
            raise ValueError(
                "Cohere response missing 'embeddings' field"
            )

        if not isinstance(vectors_list, list) or len(vectors_list) != expected_count:
            raise ValueError(
                f"Cohere returned {len(vectors_list) if isinstance(vectors_list, list) else 'invalid'} "
                f"embeddings but expected {expected_count}"
            )

        return [[float(v) for v in vec] for vec in vectors_list]


# ---------------------------------------------------------------------------
# Voyage adapter
# ---------------------------------------------------------------------------

class VoyageAdapter:
    """Voyage AI embedding API adapter.

    Endpoint: ``POST /v1/embeddings``
    Request: ``{"input": [...], "model": "..."}``
    Response: ``{"data": [{"embedding": [...], "index": N}], "model": "..."}``
    """

    def format_request(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[str, dict]:
        body: dict = {"input": texts, "model": model}
        if dimensions and dimensions > 0:
            body["dimensions"] = dimensions
        return "/v1/embeddings", body

    def parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        raw_embeddings = data.get("data")
        if not isinstance(raw_embeddings, list):
            raise ValueError(
                "Voyage response missing 'data' list"
            )
        sorted_by_index = sorted(raw_embeddings, key=lambda x: x.get("index", 0))
        vectors: list[list[float]] = []
        for item in sorted_by_index:
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise ValueError(
                    f"Voyage response item missing 'embedding' list: {item}"
                )
            vectors.append([float(v) for v in vec])

        if len(vectors) != expected_count:
            raise ValueError(
                f"Voyage returned {len(vectors)} embeddings but expected {expected_count}"
            )
        return vectors


# ---------------------------------------------------------------------------
# HuggingFace Inference API adapter
# ---------------------------------------------------------------------------

class HuggingFaceAdapter:
    """HuggingFace Inference API embedding adapter.

    Endpoint: Model-specific (base_url already includes full model path).
    Request: ``{"inputs": [...]}``
    Response: Raw nested array ``[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]``

    Note: The HuggingFace Inference API returns a raw JSON array rather than
    a structured response object.
    """

    def format_request(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[str, dict]:
        # HuggingFace endpoint is typically the full model URL.
        # The endpoint path is empty since base_url already includes the model
        # path (e.g. "https://api-inference.huggingface.co/models/<model>").
        return "", {"inputs": texts}

    def parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        # HuggingFace returns a raw array. The data may be a list directly
        # or wrapped in a dict with a key.

        # Check if data is already a list
        if isinstance(data, list):
            vectors_list = data
        else:
            # Try to find a list value in the response
            for value in data.values():
                if isinstance(value, list) and len(value) == expected_count:
                    vectors_list = value
                    break
            else:
                raise ValueError(
                    "HuggingFace response has unexpected format — expected a list of embeddings"
                )

        if len(vectors_list) != expected_count:
            raise ValueError(
                f"HuggingFace returned {len(vectors_list)} embeddings but expected {expected_count}"
            )

        return [[float(v) for v in vec] for vec in vectors_list]


# ---------------------------------------------------------------------------
# Ollama API adapter (for use through the generic provider path)
# ---------------------------------------------------------------------------

class OllamaAPIAdapter:
    """Ollama API adapter for use through the generic provider path.

    Endpoint: ``POST /api/embed``
    Request: ``{"model": "...", "input": [...]}``
    Response: ``{"embeddings": [[...], ...]}``

    Note: This adapter mimics the same request/response format used by
    ``OllamaProvider``, but goes through the generic provider path for
    consistency.
    """

    def format_request(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[str, dict]:
        body: dict = {"model": model, "input": texts}
        if dimensions and dimensions > 0:
            body["dimensions"] = dimensions
        return "/api/embed", body

    def parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        raw = data.get("embeddings")
        if not isinstance(raw, list):
            raise ValueError(
                "OllamaAPI response missing 'embeddings' list"
            )

        if len(raw) != expected_count:
            raise ValueError(
                f"OllamaAPI returned {len(raw)} embeddings but expected {expected_count}"
            )

        return [[float(v) for v in vec] for vec in raw]


# ---------------------------------------------------------------------------
# Adapter registry and selection
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, APIAdapter] = {
    "openai": OpenAIAdapter(),
    "cohere": CohereAdapter(),
    "voyage": VoyageAdapter(),
    "huggingface": HuggingFaceAdapter(),
    "ollama": OllamaAPIAdapter(),
}


def get_api_adapter(api_type: str) -> APIAdapter:
    """Return the API adapter for the given ``api_type``.

    Args:
        api_type: API type string (e.g. ``"openai"``, ``"cohere"``,
            ``"voyage"``, ``"huggingface"``, ``"ollama"``, or empty).

    Returns:
        An ``APIAdapter`` instance. Unknown types fall back to
        ``OpenAIAdapter`` with a warning log.
    """
    normalized = (api_type or "").strip().lower()
    adapter = _ADAPTER_REGISTRY.get(normalized)
    if adapter is None:
        logger.warning(
            "Unknown EMBEDDING_API_TYPE=%r, falling back to OpenAI format adapter.",
            normalized,
        )
        return OpenAIAdapter()
    return adapter
