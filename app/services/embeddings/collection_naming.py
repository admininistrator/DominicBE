"""Collection name suggestion and validation utilities.

These pure functions generate deterministic Qdrant collection names from
provider + model combinations and validate existing collection config.

Collection naming convention::

    knowledge_{provider}_{sanitized_model}

Examples:
    - ``("api", "text-embedding-3-small")`` → ``"knowledge_api_text_embedding_3_small"``
    - ``("ollama", "qwen3-embedding:0.6b")`` → ``"knowledge_ollama_qwen3_embedding_0_6b"``
    - ``("local", "local-hash-v1")`` → ``"knowledge_local_local_hash_v1"``

No dependency on CRUD, endpoints, vector_store, chat, or LlamaIndex.
"""
from __future__ import annotations

import re

# Qdrant maximum collection name length
_QDRANT_MAX_COLLECTION_LENGTH = 63

# Known legacy collection names
_LEGACY_COLLECTION = "knowledge_chunks"


def _sanitize_model_name(model: str) -> str:
    """Sanitize a model name for use in a collection name.

    Rules:
    1. Lowercase.
    2. Replace ``/``, ``:``, ``.``, ``-`` with ``_``.
    3. Strip leading and trailing underscores.
    4. Collapse multiple consecutive underscores into one.
    5. Truncate to at most 63 characters minus the prefix overhead.

    Args:
        model: Raw model name (e.g. ``"text-embedding-3-small"``).

    Returns:
        Sanitized model name safe for Qdrant collection names.
    """
    sanitized = model.lower()
    sanitized = re.sub(r"[/:.\-]", "_", sanitized)
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


def suggest_collection_name(provider: str, model: str) -> str:
    """Generate a deterministic Qdrant collection name from provider + model.

    Args:
        provider: Provider identifier (e.g. ``"local"``, ``"ollama"``, ``"api"``).
        model: Model name (e.g. ``"local-hash-v1"``, ``"text-embedding-3-small"``).

    Returns:
        A sanitized collection name suitable for Qdrant, e.g.
        ``"knowledge_api_text_embedding_3_small"``.
    """
    prefix = f"knowledge_{provider}_"
    sanitized = _sanitize_model_name(model)
    max_model_len = _QDRANT_MAX_COLLECTION_LENGTH - len(prefix)
    if max_model_len < 1:
        # If the prefix alone exceeds the limit, truncate the prefix too
        return f"knowledge_{provider[:16]}_{sanitized[:_QDRANT_MAX_COLLECTION_LENGTH - 32]}"
    truncated = sanitized[:max_model_len].rstrip("_")
    return f"{prefix}{truncated}"


def validate_collection_config(provider: str, model: str, collection: str) -> list[str]:
    """Validate that a collection name matches the expected naming convention.

    Returns a list of warning strings (empty if everything looks correct).

    Args:
        provider: Provider identifier.
        model: Model name.
        collection: Current collection name (e.g. from ``VECTOR_STORE_COLLECTION``).

    Returns:
        List of human-readable warning messages. Empty list means no warnings.
    """
    warnings: list[str] = []

    if collection == _LEGACY_COLLECTION and provider != "local":
        warnings.append(
            f"Collection {collection!r} is the legacy collection typically used with "
            f"the 'local' provider. Current provider is {provider!r}. "
            f"This may cause dimension conflicts. "
            f"Consider using: {suggest_collection_name(provider, model)!r}."
        )

    expected = suggest_collection_name(provider, model)
    if collection != expected and collection != _LEGACY_COLLECTION:
        warnings.append(
            f"Collection {collection!r} does not match the expected naming convention "
            f"for provider={provider!r} model={model!r}. "
            f"Expected: {expected!r}."
        )

    return warnings
