"""API key validation, masking, and error sanitization utilities.

WARNING: This file re-exports all security utilities from rag-core.
Do NOT edit the utilities here — modify ``rag_core.embeddings.security`` instead.
"""
from __future__ import annotations

from rag_core.embeddings.security import (  # noqa: F401
    mask_api_key,
    sanitize_error_message,
    validate_api_key,
)

__all__ = [
    "mask_api_key",
    "validate_api_key",
    "sanitize_error_message",
]
