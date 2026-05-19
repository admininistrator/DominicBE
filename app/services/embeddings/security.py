"""API key validation, masking, and error sanitization utilities.

These functions ensure that API keys are never logged or exposed in error
messages, health check responses, or diagnostics output.

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import logging

from app.services.embeddings.base import EmbeddingProviderError

logger = logging.getLogger(__name__)

# API types that strictly require a non-empty key
_KEY_REQUIRED_API_TYPES = frozenset({"openai", "cohere", "voyage"})
# API types where a key is optional but recommended
_KEY_OPTIONAL_API_TYPES = frozenset({"huggingface"})


def mask_api_key(key: str) -> str:
    """Return a masked version of an API key for safe logging/display.

    Rules:
    - If key is empty or None: returns empty string.
    - If key length <= 8: returns ``"***"``.
    - Otherwise: returns ``key[:3] + "..." + key[-4:]``.

    Args:
        key: The raw API key to mask.

    Returns:
        Masked key string safe for display.
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return key[:3] + "..." + key[-4:]


def validate_api_key(key: str, provider: str, api_type: str) -> None:
    """Validate that an API key is present when required.

    Args:
        key: The API key to validate.
        provider: Provider name (e.g. ``"api"``).
        api_type: API type (e.g. ``"openai"``, ``"cohere"``, ``"voyage"``,
            ``"huggingface"``, ``"ollama"``, or empty).

    Raises:
        EmbeddingProviderError: If the API type requires a key but none is provided.
    """
    api_type_normalized = (api_type or "").strip().lower()

    if api_type_normalized in _KEY_REQUIRED_API_TYPES and not key:
        raise EmbeddingProviderError(
            f"EMBEDDING_API_KEY is required for EMBEDDING_API_TYPE={api_type!r}",
            provider=provider,
            model="",
            category="configuration_error",
        )

    if api_type_normalized in _KEY_OPTIONAL_API_TYPES and not key:
        logger.warning(
            "EMBEDDING_API_KEY is empty for EMBEDDING_API_TYPE=%r. "
            "The HuggingFace Inference API may work without a key for public "
            "models, but authentication is recommended.",
            api_type_normalized,
        )

    # Empty api_type and "ollama" require no validation


def sanitize_error_message(msg: str, api_key: str) -> str:
    """Strip any accidental API key leakage from an error message.

    If the raw ``api_key`` appears anywhere in ``msg``, it is replaced with
    the masked version.

    Args:
        msg: The original error message (possibly containing the raw key).
        api_key: The raw API key (may be empty).

    Returns:
        Sanitized message safe for logging/display.
    """
    if not api_key or not msg:
        return msg
    if api_key in msg:
        masked = mask_api_key(api_key)
        msg = msg.replace(api_key, masked)
    return msg
