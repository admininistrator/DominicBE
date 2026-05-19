"""Ingestion pipeline protocol and canonical chunk result shape.

This module defines the contract that all ingestion pipelines must satisfy.
It has NO dependency on CRUD, vector_store, endpoints, chat, or LlamaIndex modules.
Pipelines receive normalized text and return canonical IngestionChunk objects that
are immediately usable by prepare_chunks_for_indexing() in knowledge_service.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Canonical chunk shape
# ---------------------------------------------------------------------------

@dataclass
class IngestionChunk:
    """Canonical chunk produced by any ingestion pipeline.

    All fields map directly to the dict shape expected by
    prepare_chunks_for_indexing() and create_chunks_bulk().

    Required fields:
        chunk_index: Zero-based stable position within the document.
        content:     Non-empty text content of the chunk.
        token_count: Estimated token count (chars // 4 is acceptable).

    Optional metadata fields (all additive, no new DB columns required):
        page_number:   Source page number when available (PDF, PPTX).
        section_title: Section or slide title when available.
        parser_version:  Pipeline parser version string.
        chunker_version: Pipeline chunker version string.
        extra_metadata:  Any additional key-value pairs to merge into metadata_json.
    """

    chunk_index: int
    content: str
    token_count: int

    # Optional provenance fields
    page_number: int | None = None
    section_title: str | None = None
    parser_version: str = ""
    chunker_version: str = ""
    extra_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to the dict shape expected by knowledge_service helpers.

        Merges page_number, section_title, parser_version, and chunker_version
        into metadata_json so downstream code can read them without schema changes.
        Empty version strings (default when pipeline doesn't set them) are omitted
        to let downstream setdefault logic in prepare_chunks_for_indexing() fill
        appropriate fallback values.
        """
        metadata: dict = {
            "char_count": len(self.content),
            **self.extra_metadata,
        }
        if self.parser_version:
            metadata["parser_version"] = self.parser_version
        if self.chunker_version:
            metadata["chunker_version"] = self.chunker_version
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        if self.section_title is not None:
            metadata["section_title"] = self.section_title

        return {
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": self.token_count,
            "metadata_json": metadata,
        }


# ---------------------------------------------------------------------------
# Pipeline protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class IngestionPipeline(Protocol):
    """Protocol that all ingestion pipelines must implement.

    Pipelines must NOT import CRUD, vector_store, endpoints, chat, or LlamaIndex
    persistence modules.  They receive normalized text and return canonical chunks.
    """

    def chunk_document(
        self,
        text: str,
        *,
        document_id: int | None = None,
        source_uri: str | None = None,
        title: str | None = None,
    ) -> list[IngestionChunk]:
        """Split normalized text into canonical chunks.

        Args:
            text:        Normalized document text (already whitespace-cleaned).
            document_id: Optional document ID for metadata provenance.
            source_uri:  Optional source URI / filename for metadata.
            title:       Optional document title for metadata.

        Returns:
            Ordered list of IngestionChunk objects.  May be empty for blank text.

        Raises:
            IngestionPipelineError: On any pipeline-level failure.
        """
        ...

    @property
    def pipeline_name(self) -> str:
        """Short pipeline identifier, e.g. 'custom' or 'llamaindex'."""
        ...

    @property
    def parser_version(self) -> str:
        """Parser version string, e.g. 'custom-v1' or 'llamaindex-core-v1'."""
        ...

    @property
    def chunker_version(self) -> str:
        """Chunker version string, e.g. 'custom-sentence-v1' or 'llamaindex-sentence-v1'."""
        ...


# ---------------------------------------------------------------------------
# Pipeline exception
# ---------------------------------------------------------------------------

class IngestionPipelineError(RuntimeError):
    """Raised when an ingestion pipeline encounters a non-recoverable failure.

    Attributes:
        pipeline: Short pipeline name.
        category: Failure category string (e.g. 'empty_text', 'parse_error').
    """

    def __init__(
        self,
        message: str,
        *,
        pipeline: str = "",
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.pipeline = pipeline
        self.category = category
