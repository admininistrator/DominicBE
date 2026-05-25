"""Collection name suggestion and validation utilities.

WARNING: This file re-exports all utilities from rag-core.
Do NOT edit the utilities here — modify ``rag_core.embeddings.collection_naming`` instead.
"""
from __future__ import annotations

from rag_core.embeddings.collection_naming import (  # noqa: F401
    suggest_collection_name,
    validate_collection_config,
)

__all__ = [
    "suggest_collection_name",
    "validate_collection_config",
]
