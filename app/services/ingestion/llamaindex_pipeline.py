"""LlamaIndex ingestion adapter — converts LlamaIndex nodes to canonical IngestionChunk objects.

This adapter uses llama-index-core for node splitting only.  It does NOT:
  - Write to Postgres, Qdrant, or any object storage.
  - Import CRUD, vector_store, endpoints, or chat modules.
  - Call the LLM, rerank, retrieve, or persist citations.
  - Own any persistence path.

The adapter receives normalized text (already extracted and whitespace-cleaned
by knowledge_service) and returns canonical IngestionChunk objects that are
immediately usable by prepare_chunks_for_indexing().

llama-index-core is imported lazily inside __init__ so that the module can be
imported without the package installed (the factory guards the instantiation).
"""
from __future__ import annotations

import hashlib
import logging

from app.services.ingestion.base import IngestionChunk, IngestionPipelineError

logger = logging.getLogger(__name__)

_PARSER_VERSION = "llamaindex-core-v1"
_CHUNKER_VERSION = "llamaindex-sentence-v1"


class LlamaIndexPipeline:
    """LlamaIndex-backed ingestion pipeline.

    Uses SentenceSplitter from llama-index-core to split normalized text into
    nodes, then maps each node to a canonical IngestionChunk.

    Metadata preserved per chunk:
        - chunk_index (stable, zero-based)
        - content (node text)
        - token_count (estimated from chars)
        - page_number (from node metadata when available)
        - section_title (from node metadata when available)
        - parser_version = 'llamaindex-core-v1'
        - chunker_version = 'llamaindex-sentence-v1'
        - extra_metadata: any additional node metadata keys
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialise the pipeline and verify llama-index-core is available.

        Args:
            chunk_size:    Characters per chunk (defaults to settings.chunk_size).
            chunk_overlap: Overlap characters (defaults to settings.chunk_overlap).

        Raises:
            IngestionPipelineError: If llama-index-core is not installed.
        """
        try:
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core.schema import Document as LlamaDocument
        except ImportError as exc:
            raise IngestionPipelineError(
                f"LlamaIndexPipeline requires llama-index-core. "
                f"Install with: pip install llama-index-core. "
                f"Original error: {exc}",
                pipeline="llamaindex",
                category="missing_dependency",
            ) from exc

        from app.core.config import settings

        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._SentenceSplitter = SentenceSplitter
        self._LlamaDocument = LlamaDocument

        logger.debug(
            "LlamaIndexPipeline: chunk_size=%d chunk_overlap=%d",
            self._chunk_size,
            self._chunk_overlap,
        )

    # IngestionPipeline protocol properties
    @property
    def pipeline_name(self) -> str:
        return "llamaindex"

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    @property
    def chunker_version(self) -> str:
        return _CHUNKER_VERSION

    def chunk_document(
        self,
        text: str,
        *,
        document_id: int | None = None,
        source_uri: str | None = None,
        title: str | None = None,
    ) -> list[IngestionChunk]:
        """Split normalized text into canonical chunks using LlamaIndex SentenceSplitter.

        Args:
            text:        Normalized document text.
            document_id: Optional document ID for logging.
            source_uri:  Optional source URI stored in node metadata.
            title:       Optional document title stored in node metadata.

        Returns:
            Ordered list of IngestionChunk objects.  Empty list for blank text.

        Raises:
            IngestionPipelineError: On any LlamaIndex-level failure.
        """
        if not (text or "").strip():
            return []

        try:
            return self._split(text, document_id=document_id, source_uri=source_uri, title=title)
        except IngestionPipelineError:
            raise
        except Exception as exc:
            raise IngestionPipelineError(
                f"LlamaIndexPipeline: node splitting failed for document_id={document_id}: {exc}",
                pipeline=self.pipeline_name,
                category="parse_error",
            ) from exc

    def _split(
        self,
        text: str,
        *,
        document_id: int | None,
        source_uri: str | None,
        title: str | None,
    ) -> list[IngestionChunk]:
        """Internal split implementation — separated for testability."""
        # Build a LlamaIndex Document with document-level metadata.
        # We do NOT pass embedding or LLM — this adapter is chunking-only.
        doc_metadata: dict = {}
        if source_uri:
            doc_metadata["source_uri"] = source_uri
        if title:
            doc_metadata["title"] = title
        if document_id is not None:
            doc_metadata["document_id"] = document_id

        llama_doc = self._LlamaDocument(
            text=text,
            metadata=doc_metadata,
        )

        # SentenceSplitter: chunk_size in tokens (approx chars // 4 for English).
        # We pass chunk_size in characters and let LlamaIndex use its default
        # tokenizer (whitespace-based) which is close enough for our purposes.
        splitter = self._SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents([llama_doc])

        chunks: list[IngestionChunk] = []
        for idx, node in enumerate(nodes):
            node_text = (node.get_content() or "").strip()
            if not node_text:
                continue

            # Extract node-level metadata — LlamaIndex stores these in node.metadata
            node_meta: dict = dict(node.metadata or {})

            # Remove document-level keys we set ourselves (avoid duplication)
            node_meta.pop("source_uri", None)
            node_meta.pop("title", None)
            node_meta.pop("document_id", None)

            # Extract well-known optional fields
            page_number: int | None = None
            raw_page = node_meta.pop("page_label", None) or node_meta.pop("page_number", None)
            if raw_page is not None:
                try:
                    page_number = int(raw_page)
                except (TypeError, ValueError):
                    pass

            section_title: str | None = node_meta.pop("section_title", None) or node_meta.pop(
                "header", None
            )

            chunks.append(
                IngestionChunk(
                    chunk_index=idx,
                    content=node_text,
                    token_count=max(1, len(node_text) // 4),
                    page_number=page_number,
                    section_title=section_title,
                    parser_version=_PARSER_VERSION,
                    chunker_version=_CHUNKER_VERSION,
                    extra_metadata=node_meta,
                )
            )

        logger.debug(
            "LlamaIndexPipeline: document_id=%s produced %d chunks from %d nodes",
            document_id,
            len(chunks),
            len(nodes),
        )
        return chunks
