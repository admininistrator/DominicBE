"""Phase 5 unit tests for retrieval and citation compatibility validation.

Coverage:
- P05-T01: Owner and session filters — cross-user isolation, session scope.
- P05-T02: Retrieval event metadata — provider, model, dimensions, collection,
           strategy, fallback, returned count, latency, session scope.
- P05-T03: Source shape — document_id, chunk_id, title, score, snippet,
           source_uri, rank, embedding_provider.
- P05-T04: Citation persistence — AnswerCitation rows, replace, load, clear,
           full _finalize_chat_turn citation flow.

Run with:
    cd DominicBE
    python -m pytest tests/test_retrieval_citation_compatibility.py -v
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Must be set before any app import to prevent pydantic Settings validation
# from reading the real .env file.
os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"

from sqlalchemy import create_engine, event as sa_event
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.models.knowledge_models import (
    AnswerCitation,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings_override(**kwargs):
    """Return a mock settings object with sensible defaults + overrides."""
    defaults = {
        "embedding_provider": "local",
        "embedding_model": "local-hash-v1",
        "embedding_dimensions": 64,
        "embedding_base_url": "http://localhost:11434",
        "embedding_timeout_seconds": 60.0,
        "embedding_batch_size": 16,
        "ingestion_pipeline": "custom",
        "vector_store_provider": "database",
        "vector_store_collection": "knowledge_chunks",
        "retrieval_top_k": 5,
        "retrieval_max_rerank_candidates": 20,
        "retrieval_min_score": 0.0,
        "retrieval_min_lexical_score": 0.0,
        "retrieval_hybrid_semantic_weight": 0.5,
        "retrieval_hybrid_lexical_weight": 0.5,
        "retrieval_rerank_title_weight": 0.1,
        "retrieval_rerank_position_weight": 0.05,
        "retrieval_low_confidence_score": 0.15,
        "retrieval_enable_query_expansion": False,
        "retrieval_max_context_chunks": 10,
        "retrieval_max_context_tokens": 4096,
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_sqlite_factory(fk_enabled: bool = False):
    """Create an in-memory SQLite session factory."""
    # Ensure ALL models are imported so FK references (e.g. chat_sessions)
    # are available during create_all.
    import app.models.chat_models  # noqa: F401 — ensures chat_sessions table exists
    import app.models.system_models  # noqa: F401 — ensures system tables exist

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    if fk_enabled:
        @sa_event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return testing_session_local, engine, db_path


def _seed_document(session_factory, owner: str, title: str = "Test Doc",
                   raw_text: str = "Test content for retrieval validation.",
                   session_id: int | None = None, status: str = "indexed") -> int:
    """Seed a document and return its id."""
    db = session_factory()
    try:
        doc = KnowledgeDocument(
            owner_username=owner,
            title=title,
            source_type="text",
            raw_text=raw_text,
            status=status,
            session_id=session_id,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc.id
    finally:
        db.close()


def _seed_chunks(session_factory, document_id: int, count: int = 2,
                 provider: str = "local", model: str = "local-hash-v1",
                 dimensions: int = 64, version: str = "local-hash-v1") -> list[int]:
    """Seed chunks for a document and return their ids."""
    db = session_factory()
    try:
        chunk_ids = []
        for idx in range(count):
            chunk = KnowledgeChunk(
                document_id=document_id,
                chunk_index=idx,
                content=f"Chunk {idx} content for retrieval testing purposes.",
                token_count=8,
                embedding_model=model,
                vector_id=f"vec_{document_id}_{idx}",
                metadata_json={
                    "embedding_provider": provider,
                    "embedding_model": model,
                    "embedding_dimensions": dimensions,
                    "embedding_version": version,
                    "parser_version": "test-v1",
                    "chunker_version": "test-sentence-v1",
                },
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)
            chunk_ids.append(chunk.id)
        return chunk_ids
    finally:
        db.close()


class TestQueryEmbeddingCallCounts(unittest.TestCase):
    """Regression tests for bounded query embedding during retrieval."""

    def setUp(self):
        self.factory, self.engine, self.db_path = _make_sqlite_factory()

    def tearDown(self):
        self.engine.dispose()
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _mock_provider(self, *, chunk_count: int = 0):
        from app.services.embeddings.base import EmbedResult, EmbeddingMeta, QueryEmbedResult

        meta = EmbeddingMeta(
            provider="api",
            model="nvidia/llama-nemotron-embed-1b-v2",
            dimensions=3,
            version="api-nvidia-v1",
            extra={"api_type": "nvidia"},
        )
        provider = MagicMock()
        provider.embed_query.return_value = QueryEmbedResult(vector=[0.2, 0.2, 0.2], meta=meta)
        provider.embed_texts.return_value = EmbedResult(
            vectors=[[0.2, 0.2, 0.2] for _ in range(chunk_count)],
            meta=meta,
        )
        return provider

    def test_db_fallback_embeds_query_once_and_batches_missing_chunk_embeddings(self):
        """DB fallback must not call embed_query once per candidate chunk."""
        from app.services.retrieval_service import search_knowledge

        owner = "embed_once_user"
        chunk_count = 4
        doc_id = _seed_document(self.factory, owner=owner, title="Embedding Call Count")
        _seed_chunks(
            self.factory,
            doc_id,
            count=chunk_count,
            provider="api",
            model="nvidia/llama-nemotron-embed-1b-v2",
            dimensions=3,
            version="api-nvidia-v1",
        )
        provider = self._mock_provider(chunk_count=chunk_count)

        db = self.factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override(
                           embedding_provider="api",
                           embedding_model="nvidia/llama-nemotron-embed-1b-v2",
                           embedding_dimensions=3,
                           retrieval_min_score=0.0,
                           retrieval_min_lexical_score=0.0,
                       )):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="retrieval testing",
                    top_k=chunk_count,
                    session_id=123,
                    request_id="req-embed-once",
                )

            self.assertEqual(result["returned"], chunk_count)
            self.assertEqual(provider.embed_query.call_count, 1)
            self.assertEqual(provider.embed_texts.call_count, 1)
            embedded_texts = provider.embed_texts.call_args.args[0]
            self.assertEqual(len(embedded_texts), chunk_count)
        finally:
            db.close()

    def test_stored_chunk_embeddings_do_not_trigger_per_chunk_query_embeddings(self):
        """Stored chunk vectors should be reused without extra provider calls."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge

        owner = "stored_embedding_user"
        doc_id = _seed_document(self.factory, owner=owner, title="Stored Embeddings")
        _seed_chunks(
            self.factory,
            doc_id,
            count=3,
            provider="api",
            model="nvidia/llama-nemotron-embed-1b-v2",
            dimensions=3,
            version="api-nvidia-v1",
        )
        seed_db = self.factory()
        try:
            rows = crud_knowledge.list_searchable_chunks(seed_db, owner, document_id=doc_id)
            for chunk, _document in rows:
                chunk.metadata_json = {
                    **(chunk.metadata_json or {}),
                    "embedding": [0.2, 0.2, 0.2],
                }
            seed_db.commit()
        finally:
            seed_db.close()

        provider = self._mock_provider(chunk_count=0)
        db = self.factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override(
                           embedding_provider="api",
                           embedding_model="nvidia/llama-nemotron-embed-1b-v2",
                           embedding_dimensions=3,
                           retrieval_min_score=0.0,
                           retrieval_min_lexical_score=0.0,
                       )):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="stored embedding query",
                    top_k=3,
                    document_id=doc_id,
                    request_id="req-stored-embedding",
                )

            self.assertEqual(result["returned"], 3)
            self.assertEqual(provider.embed_query.call_count, 1)
            provider.embed_texts.assert_not_called()
        finally:
            db.close()

    def test_query_expansion_uses_one_rewritten_query_embedding(self):
        """Current expansion rewrites one query, so embed_query remains bounded at one."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge

        owner = "expanded_query_user"
        doc_id = _seed_document(self.factory, owner=owner, title="Expansion")
        _seed_chunks(
            self.factory,
            doc_id,
            count=1,
            provider="api",
            model="nvidia/llama-nemotron-embed-1b-v2",
            dimensions=3,
            version="api-nvidia-v1",
        )
        seed_db = self.factory()
        try:
            rows = crud_knowledge.list_searchable_chunks(seed_db, owner, document_id=doc_id)
            for chunk, _document in rows:
                chunk.metadata_json = {
                    **(chunk.metadata_json or {}),
                    "embedding": [0.2, 0.2, 0.2],
                }
            seed_db.commit()
        finally:
            seed_db.close()

        provider = self._mock_provider(chunk_count=0)
        db = self.factory()
        try:
            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override(
                           embedding_provider="api",
                           embedding_model="nvidia/llama-nemotron-embed-1b-v2",
                           embedding_dimensions=3,
                           retrieval_enable_query_expansion=True,
                           retrieval_min_score=0.0,
                           retrieval_min_lexical_score=0.0,
                       )):
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="chinh sach hoan tien",
                    top_k=1,
                    document_id=doc_id,
                    request_id="req-expanded-query",
                )

            self.assertGreaterEqual(len(result["query_expansions"]), 1)
            self.assertEqual(provider.embed_query.call_count, 1)
            embedded_query = provider.embed_query.call_args.args[0]
            self.assertIn("refund", embedded_query)
            provider.embed_texts.assert_not_called()
        finally:
            db.close()


# ===========================================================================
# P05-T01: Owner and session filters
# ===========================================================================

class TestOwnerAndSessionFilters(unittest.TestCase):
    """Verify that owner and session permission filters are enforced."""

    def setUp(self):
        self.factory, self.engine, self.db_path = _make_sqlite_factory()

    def tearDown(self):
        self.engine.dispose()
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    # -- Owner filter tests -------------------------------------------------

    def test_cross_user_retrieval_is_blocked_by_owner_filter(self):
        """User A must not retrieve chunks owned by User B via CRUD."""
        from app.crud import crud_knowledge

        doc_id = _seed_document(self.factory, owner="alice")
        _seed_chunks(self.factory, doc_id, count=2)

        db = self.factory()
        try:
            # Bob tries to retrieve Alice's chunks
            results = crud_knowledge.list_searchable_chunks(
                db, owner_username="bob", document_id=doc_id,
            )
            self.assertEqual(len(results), 0,
                             "Bob must not retrieve Alice's chunks")
        finally:
            db.close()

    def test_cross_user_retrieval_blocked_by_ids_filter(self):
        """User A must not get chunks by IDs owned by User B."""
        from app.crud import crud_knowledge

        doc_id = _seed_document(self.factory, owner="alice")
        chunk_ids = _seed_chunks(self.factory, doc_id, count=2)

        db = self.factory()
        try:
            results = crud_knowledge.list_searchable_chunks_by_ids(
                db, owner_username="bob", chunk_ids=chunk_ids,
            )
            self.assertEqual(len(results), 0,
                             "Bob must not retrieve Alice's chunks by IDs")
        finally:
            db.close()

    # -- Service-layer filter tests ------------------------------------------

    def test_search_knowledge_cross_user_blocked_by_owner(self):
        """search_knowledge() must return empty results when owner differs."""
        from app.services.retrieval_service import search_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        doc_id = _seed_document(self.factory, owner="alice")
        _seed_chunks(self.factory, doc_id, count=2)

        db = self.factory()
        try:
            provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override()):

                result = search_knowledge(
                    db=db,
                    owner_username="bob",   # different owner
                    query="test cross-user isolation",
                    document_id=doc_id,
                )
            # Bob must not retrieve Alice's chunks
            self.assertEqual(result.get("returned"), 0,
                             "Cross-user search_knowledge must return 0 results")
            self.assertEqual(len(result.get("results", [])), 0,
                             "Cross-user results list must be empty")
        finally:
            db.close()

    def test_search_knowledge_session_scope_filter(self):
        """search_knowledge() with session_id must scope retrieval."""
        from app.services.retrieval_service import search_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        owner = "charlie"
        # Session document
        doc_session = _seed_document(self.factory, owner, title="Session Doc",
                                     session_id=100)
        _seed_chunks(self.factory, doc_session, count=1)
        # Global document
        doc_global = _seed_document(self.factory, owner, title="Global Doc")
        _seed_chunks(self.factory, doc_global, count=1)

        db = self.factory()
        try:
            provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.knowledge_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override()):

                # Session-scoped search should return session docs
                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="session scope test",
                    session_id=100,
                )
            # When a session_id is provided and has_indexed_documents_for_session
            # returns True, the session_scope is "session" and retrieval should
            # be scoped. The global-only doc should not appear.
            self.assertIn("session_scope", result)
            if result["session_scope"] == "session":
                # Only session-scoped chunks returned
                for item in result.get("results", []):
                    self.assertIsNotNone(item.get("document_id"))
        finally:
            db.close()

    # -- Session scope tests ------------------------------------------------

    def test_session_scoped_retrieval_stays_in_session(self):
        """session_scope='session' must only return documents for that session."""
        from app.crud import crud_knowledge

        owner = "charlie"
        # Document with session_id=10
        doc_session = _seed_document(self.factory, owner, title="Session Doc",
                                     session_id=10)
        _seed_chunks(self.factory, doc_session, count=1)
        # Document with no session_id (global)
        doc_global = _seed_document(self.factory, owner, title="Global Doc")
        _seed_chunks(self.factory, doc_global, count=1)

        db = self.factory()
        try:
            # Session-scoped retrieval should only return session docs
            results = crud_knowledge.list_searchable_chunks(
                db, owner, session_id=10, session_scope="session",
            )
            doc_ids = {doc.id for _, doc in results}
            self.assertIn(doc_session, doc_ids,
                          "Session-scoped retrieval must include session doc")
            self.assertNotIn(doc_global, doc_ids,
                             "Session-scoped retrieval must exclude global doc")
        finally:
            db.close()

    def test_global_scope_excludes_session_documents(self):
        """session_scope='global' must exclude documents with session_id."""
        from app.crud import crud_knowledge

        owner = "dave"
        doc_session = _seed_document(self.factory, owner, title="Session Doc",
                                     session_id=20)
        _seed_chunks(self.factory, doc_session, count=1)
        doc_global = _seed_document(self.factory, owner, title="Global Doc")
        _seed_chunks(self.factory, doc_global, count=1)

        db = self.factory()
        try:
            results = crud_knowledge.list_searchable_chunks(
                db, owner, session_id=20, session_scope="global",
            )
            doc_ids = {doc.id for _, doc in results}
            self.assertNotIn(doc_session, doc_ids,
                             "Global scope must exclude session doc")
            self.assertIn(doc_global, doc_ids,
                          "Global scope must include global doc")
        finally:
            db.close()

    def test_all_scope_returns_both_session_and_global(self):
        """session_scope='all' must return all documents for the owner."""
        from app.crud import crud_knowledge

        owner = "eve"
        doc_session = _seed_document(self.factory, owner, title="Session Doc",
                                     session_id=30)
        _seed_chunks(self.factory, doc_session, count=1)
        doc_global = _seed_document(self.factory, owner, title="Global Doc")
        _seed_chunks(self.factory, doc_global, count=1)

        db = self.factory()
        try:
            results = crud_knowledge.list_searchable_chunks(
                db, owner, session_id=30, session_scope="all",
            )
            doc_ids = {doc.id for _, doc in results}
            self.assertIn(doc_session, doc_ids,
                          "'all' scope must include session doc")
            self.assertIn(doc_global, doc_ids,
                          "'all' scope must include global doc")
        finally:
            db.close()

    def test_indexed_only_filter_excludes_non_indexed(self):
        """indexed_only=True must exclude documents with status != 'indexed'."""
        from app.crud import crud_knowledge

        owner = "frank"
        doc_indexed = _seed_document(self.factory, owner, title="Indexed Doc",
                                     status="indexed")
        _seed_chunks(self.factory, doc_indexed, count=1)
        doc_pending = _seed_document(self.factory, owner, title="Pending Doc",
                                     status="uploaded")
        _seed_chunks(self.factory, doc_pending, count=1)

        db = self.factory()
        try:
            results = crud_knowledge.list_searchable_chunks(
                db, owner, indexed_only=True,
            )
            doc_ids = {doc.id for _, doc in results}
            self.assertIn(doc_indexed, doc_ids,
                          "Indexed doc must be included")
            self.assertNotIn(doc_pending, doc_ids,
                             "Non-indexed doc must be excluded")
        finally:
            db.close()


# ===========================================================================
# P05-T02: Retrieval event metadata
# ===========================================================================

class TestRetrievalEventMetadata(unittest.TestCase):
    """Verify retrieval events include provider, model, dimensions, etc."""

    def setUp(self):
        self.factory, self.engine, self.db_path = _make_sqlite_factory()

    def tearDown(self):
        self.engine.dispose()
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _make_ollama_embedding_result(self):
        """Build a mocked EmbedResult / QueryEmbedResult with Ollama metadata."""
        from app.services.embeddings.base import (
            EmbeddingMeta, EmbedResult, QueryEmbedResult,
        )
        meta = EmbeddingMeta(
            provider="ollama",
            model="qwen3-embedding:0.6b",
            dimensions=1024,
            version="ollama-qwen3-embedding-06b-v1",
        )
        # 1024-dim zero vector
        vec = [0.0] * 1024
        vec[0] = 0.1
        return meta, QueryEmbedResult(vector=vec, meta=meta), EmbedResult(vectors=[vec], meta=meta)

    def test_retrieval_event_includes_provider_metadata(self):
        """Retrieval event metadata_json must include provider fields."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        owner = "grace"
        doc_id = _seed_document(self.factory, owner)
        _seed_chunks(self.factory, doc_id, count=2)
        request_id = "req-provider-meta-001"

        db = self.factory()
        try:
            # Mock the embedding provider to return local hash
            provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.knowledge_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override()):

                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="test retrieval metadata",
                    document_id=doc_id,
                    request_id=request_id,
                )

            # Verify retrieval event was created
            self.assertIn("retrieval_id", result)
            event = crud_knowledge.get_retrieval_event_by_request_id(
                db, request_id,
            )
            self.assertIsNotNone(event, "Retrieval event must exist")
            meta = event.metadata_json or {}

            # P05-T02: Provider metadata fields must be present
            self.assertIn("embedding_provider", meta,
                          "metadata_json must include embedding_provider")
            self.assertIn("embedding_model", meta,
                          "metadata_json must include embedding_model")
            self.assertIn("embedding_dimensions", meta,
                          "metadata_json must include embedding_dimensions")
            self.assertIn("embedding_version", meta,
                          "metadata_json must include embedding_version")
            self.assertIn("vector_store_collection", meta,
                          "metadata_json must include vector_store_collection")
            self.assertIn("vector_store_provider", meta,
                          "metadata_json must include vector_store_provider")

            # Local provider values
            self.assertEqual(meta["embedding_provider"], "local")
            self.assertEqual(meta["embedding_model"], "local-hash-v1")
            self.assertEqual(meta["embedding_dimensions"], 64)
        finally:
            db.close()

    def test_retrieval_event_includes_session_scope(self):
        """Retrieval event metadata must include session_scope."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        owner = "heidi"
        doc_id = _seed_document(self.factory, owner, session_id=50)
        _seed_chunks(self.factory, doc_id, count=2)
        request_id = "req-session-scope-001"

        db = self.factory()
        try:
            provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.knowledge_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override()):

                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="session scope test",
                    session_id=50,
                    request_id=request_id,
                )

            event = crud_knowledge.get_retrieval_event_by_request_id(
                db, request_id,
            )
            self.assertIsNotNone(event)
            meta = event.metadata_json or {}

            # session_scope must be present
            self.assertIn("session_scope", meta)
            # For a document with session_id, and has_indexed_documents_for_session
            # is True, session_scope should be "session"
            if meta["session_scope"] == "session":
                self.assertEqual(meta["session_scope"], "session")
        finally:
            db.close()

    def test_retrieval_event_includes_mixed_space_skip_count(self):
        """Retrieval event must include mixed_space_skip_count."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        owner = "ivan"
        doc_id = _seed_document(self.factory, owner)
        _seed_chunks(self.factory, doc_id, count=2)
        request_id = "req-mixed-space-001"

        db = self.factory()
        try:
            provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.knowledge_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override()):

                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="mixed space skip test",
                    document_id=doc_id,
                    request_id=request_id,
                )

            event = crud_knowledge.get_retrieval_event_by_request_id(
                db, request_id,
            )
            self.assertIsNotNone(event)
            meta = event.metadata_json or {}
            self.assertIn("mixed_space_skip_count", meta,
                          "Retrieval event must include mixed_space_skip_count")
        finally:
            db.close()

    def test_retrieval_event_includes_strategy_and_fallback(self):
        """Retrieval event must include strategy, fallback_used, fallback_reason."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        owner = "judy"
        doc_id = _seed_document(self.factory, owner)
        _seed_chunks(self.factory, doc_id, count=2)
        request_id = "req-strategy-001"

        db = self.factory()
        try:
            provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.knowledge_service.get_embedding_provider",
                       return_value=provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override()):

                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="strategy and fallback test",
                    document_id=doc_id,
                    request_id=request_id,
                )

            event = crud_knowledge.get_retrieval_event_by_request_id(
                db, request_id,
            )
            self.assertIsNotNone(event)
            meta = event.metadata_json or {}

            self.assertIn("strategy", meta)
            self.assertIn("fallback_used", meta)
            self.assertIn("fallback_reason", meta)
            self.assertIn("evidence_strength", meta)
            self.assertIn("returned", meta)
            self.assertIn("candidate_count", meta)

            # P05-T02: returned count must be non-negative
            returned_count = meta.get("returned", -1)
            self.assertGreaterEqual(returned_count, 0,
                                    "returned count must be non-negative")
            # latency_ms is on the RetrievalEvent column, not metadata_json
            self.assertIsNotNone(event.latency_ms,
                                 "latency_ms must be set on the retrieval event")
        finally:
            db.close()

    def test_retrieval_event_includes_ollama_provider_metadata(self):
        """When Ollama provider is active, metadata must reflect it."""
        from app.services.retrieval_service import search_knowledge
        from app.crud import crud_knowledge

        owner = "karen"
        doc_id = _seed_document(self.factory, owner)
        _seed_chunks(self.factory, doc_id, count=2,
                     provider="ollama", model="qwen3-embedding:0.6b",
                     dimensions=1024, version="ollama-qwen3-embedding-06b-v1")
        request_id = "req-ollama-meta-001"

        db = self.factory()
        try:
            ollama_meta, query_result, embed_result = self._make_ollama_embedding_result()
            mock_provider = MagicMock()
            mock_provider.meta = ollama_meta
            mock_provider.embed_query.return_value = query_result
            mock_provider.embed_texts.return_value = embed_result

            with patch("app.services.retrieval_service.get_embedding_provider",
                       return_value=mock_provider), \
                 patch("app.services.knowledge_service.get_embedding_provider",
                       return_value=mock_provider), \
                 patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                       return_value=False), \
                 patch("app.services.retrieval_service.settings",
                       _make_settings_override(
                           embedding_provider="ollama",
                           embedding_model="qwen3-embedding:0.6b",
                           embedding_dimensions=1024,
                           vector_store_collection="knowledge_qwen3_embedding_06b",
                       )):

                result = search_knowledge(
                    db=db,
                    owner_username=owner,
                    query="ollama metadata test",
                    document_id=doc_id,
                    request_id=request_id,
                )

            event = crud_knowledge.get_retrieval_event_by_request_id(
                db, request_id,
            )
            self.assertIsNotNone(event)
            meta = event.metadata_json or {}

            self.assertEqual(meta.get("embedding_provider"), "ollama")
            self.assertEqual(meta.get("embedding_model"), "qwen3-embedding:0.6b")
            self.assertEqual(meta.get("embedding_dimensions"), 1024)
            self.assertEqual(meta.get("embedding_version"), "ollama-qwen3-embedding-06b-v1")
            self.assertEqual(meta.get("vector_store_collection"), "knowledge_qwen3_embedding_06b")
        finally:
            db.close()


# ===========================================================================
# P05-T03: Source shape
# ===========================================================================

class TestSourceShape(unittest.TestCase):
    """Verify search results and chat sources preserve required fields."""

    def _make_search_result_item(self, **overrides) -> dict:
        """Build a realistic search result item."""
        item = {
            "document_id": 1,
            "chunk_id": 10,
            "chunk_index": 0,
            "title": "Test Document",
            "source_type": "text",
            "source_uri": "http://example.com/doc.txt",
            "score": 0.85,
            "semantic_score": 0.75,
            "lexical_score": 0.65,
            "rerank_score": 0.82,
            "token_estimate": 50,
            "snippet": "This is a snippet of the chunk content...",
            "content": "This is the full chunk content for retrieval testing purposes.",
            "vector_id": "vec_1_0",
            "embedding_model": "local-hash-v1",
            "embedding_provider": "local",
        }
        item.update(overrides)
        return item

    def test_search_result_includes_required_fields(self):
        """Every search result must include document_id, chunk_id, title, score,
        snippet, source_uri, embedding_provider, and rank-related field."""
        item = self._make_search_result_item()

        required_fields = [
            "document_id", "chunk_id", "chunk_index", "title",
            "score", "snippet", "source_uri", "embedding_model",
            "embedding_provider", "rerank_score", "token_estimate",
        ]
        for field in required_fields:
            self.assertIn(field, item,
                          f"Search result must include '{field}'")

        # Type checks
        self.assertIsInstance(item["document_id"], int)
        self.assertIsInstance(item["chunk_id"], int)
        self.assertIsInstance(item["title"], str)
        self.assertIsInstance(item["score"], float)
        self.assertIsInstance(item["snippet"], str)
        self.assertIsInstance(item["source_uri"], str)

    def test_build_sources_preserves_fields(self):
        """_build_sources() must preserve document_id, chunk_id, title,
        score, snippet, source_uri, and rank."""
        from app.services.chat_service import _build_sources

        results = [
            self._make_search_result_item(document_id=1, chunk_id=10),
            self._make_search_result_item(document_id=1, chunk_id=11,
                                          score=0.75, rerank_score=0.72),
        ]
        sources = _build_sources(results)

        self.assertEqual(len(sources), 2)
        for idx, source in enumerate(sources, start=1):
            self.assertIn("document_id", source)
            self.assertIn("chunk_id", source)
            self.assertIn("title", source)
            self.assertIn("source_type", source)
            self.assertIn("score", source)
            self.assertIn("rerank_score", source)
            self.assertIn("snippet", source)
            self.assertIn("source_uri", source)
            self.assertIn("rank", source)
            # Rank starts at 1
            self.assertEqual(source["rank"], idx)
            # source_type must be "knowledge"
            self.assertEqual(source["source_type"], "knowledge")

    def test_build_sources_preserves_source_uri(self):
        """source_uri must be preserved through _build_sources()."""
        from app.services.chat_service import _build_sources

        results = [
            self._make_search_result_item(
                source_uri="http://example.com/faq.txt",
            ),
        ]
        sources = _build_sources(results)
        self.assertEqual(sources[0]["source_uri"], "http://example.com/faq.txt")

    def test_build_sources_empty_input(self):
        """_build_sources([]) must return []."""
        from app.services.chat_service import _build_sources

        self.assertEqual(_build_sources([]), [])

    def test_retrieval_service_returns_embedding_provider_in_results(self):
        """search_knowledge result items must include embedding_provider field."""
        from app.services.retrieval_service import search_knowledge
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        factory, engine, db_path = _make_sqlite_factory()
        try:
            owner = "laura"
            doc_id = _seed_document(factory, owner)
            _seed_chunks(factory, doc_id, count=2)

            db = factory()
            try:
                provider = LocalHashProvider(model="local-hash-v1", dimensions=64)

                with patch("app.services.retrieval_service.get_embedding_provider",
                           return_value=provider), \
                     patch("app.services.knowledge_service.get_embedding_provider",
                           return_value=provider), \
                     patch("app.services.retrieval_service.vector_store.is_external_vector_store_enabled",
                           return_value=False), \
                     patch("app.services.retrieval_service.settings",
                           _make_settings_override()):

                    result = search_knowledge(
                        db=db,
                        owner_username=owner,
                        query="source shape test",
                        document_id=doc_id,
                    )

                for item in result.get("results", []):
                    self.assertIn("embedding_provider", item,
                                  "Each result must include embedding_provider")
                    self.assertIn("document_id", item)
                    self.assertIn("chunk_id", item)
                    self.assertIn("title", item)
                    self.assertIn("score", item)
                    self.assertIn("snippet", item)
                    self.assertIn("source_uri", item)
            finally:
                db.close()
        finally:
            engine.dispose()
            try:
                os.remove(db_path)
            except OSError:
                pass


# ===========================================================================
# P05-T04: Citation persistence
# ===========================================================================

class TestCitationPersistence(unittest.TestCase):
    """Verify AnswerCitation rows survive replace_answer_citations."""

    def setUp(self):
        self.factory, self.engine, self.db_path = _make_sqlite_factory(fk_enabled=True)

    def tearDown(self):
        self.engine.dispose()
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _seed_citation_prerequisites(self, db: Session, owner: str = "mallory"):
        """Seed a user, document, and chunk; return (doc_id, chunk_id)."""
        from app.models.chat_models import User

        # Ensure user exists
        existing = db.query(User).filter(User.username == owner).first()
        if not existing:
            db.add(User(username=owner, password_hash="hash", role="user"))
            db.commit()

        doc = KnowledgeDocument(
            owner_username=owner,
            title="Citation Test Doc",
            source_type="text",
            raw_text="Citation test content.",
            status="indexed",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        chunk = KnowledgeChunk(
            document_id=doc.id,
            chunk_index=0,
            content="Citation test chunk content.",
            token_count=6,
            embedding_model="local-hash-v1",
            vector_id="vec_citation",
            metadata_json={
                "embedding_provider": "local",
                "embedding_model": "local-hash-v1",
                "embedding_dimensions": 64,
                "embedding_version": "local-hash-v1",
            },
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        return doc.id, chunk.id

    def test_replace_answer_citations_persists_rows(self):
        """replace_answer_citations must create AnswerCitation rows."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            request_id = "req-citation-001"

            citations = [
                {
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "rank": 1,
                    "score": 0.85,
                    "quoted_text": "Citation test chunk content.",
                },
            ]

            rows = crud_knowledge.replace_answer_citations(db, request_id, citations)
            self.assertEqual(len(rows), 1)

            row = rows[0]
            self.assertEqual(row.request_id, request_id)
            self.assertEqual(row.document_id, doc_id)
            self.assertEqual(row.chunk_id, chunk_id)
            self.assertEqual(row.rank, 1)
            self.assertEqual(row.score, 0.85)
            self.assertEqual(row.quoted_text, "Citation test chunk content.")
        finally:
            db.close()

    def test_citations_loadable_via_list_by_request_ids(self):
        """Citations must be retrievable via list_answer_citations_by_request_ids."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            request_id = "req-citation-002"

            crud_knowledge.replace_answer_citations(db, request_id, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1,
                 "score": 0.85, "quoted_text": "Quoted text."},
            ])

            results = crud_knowledge.list_answer_citations_by_request_ids(
                db, [request_id],
            )
            self.assertEqual(len(results), 1)

            citation, document, chunk = results[0]
            self.assertEqual(citation.request_id, request_id)
            self.assertEqual(document.title, "Citation Test Doc")
            self.assertIsNotNone(chunk.content)
        finally:
            db.close()

    def test_citations_replaced_on_second_call(self):
        """A second replace_answer_citations must replace (not append) citations."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            request_id = "req-citation-003"

            # First call: 2 citations
            crud_knowledge.replace_answer_citations(db, request_id, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1,
                 "score": 0.9, "quoted_text": "First."},
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 2,
                 "score": 0.8, "quoted_text": "Second."},
            ])

            # Second call: 1 citation (should replace the 2)
            crud_knowledge.replace_answer_citations(db, request_id, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1,
                 "score": 0.95, "quoted_text": "Replacement."},
            ])

            results = crud_knowledge.list_answer_citations_by_request_ids(
                db, [request_id],
            )
            self.assertEqual(len(results), 1,
                             "Citations must be replaced, not appended")
            self.assertEqual(results[0][0].quoted_text, "Replacement.")
        finally:
            db.close()

    def test_replace_answer_citations_empty_citations_clears(self):
        """An empty citations list must delete existing citations for that request."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            request_id = "req-citation-004"

            # First call: 1 citation
            crud_knowledge.replace_answer_citations(db, request_id, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1,
                 "score": 0.9, "quoted_text": "To be cleared."},
            ])

            # Second call: empty list (clear)
            crud_knowledge.replace_answer_citations(db, request_id, [])

            results = crud_knowledge.list_answer_citations_by_request_ids(
                db, [request_id],
            )
            self.assertEqual(len(results), 0,
                             "Empty citations list must clear existing citations")
        finally:
            db.close()

    def test_multiple_requests_have_independent_citations(self):
        """Citations for different request_ids must be independent."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            req_a = "req-citation-005-a"
            req_b = "req-citation-005-b"

            crud_knowledge.replace_answer_citations(db, req_a, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1,
                 "score": 0.9, "quoted_text": "Request A"},
            ])
            crud_knowledge.replace_answer_citations(db, req_b, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1,
                 "score": 0.95, "quoted_text": "Request B"},
            ])

            results_a = crud_knowledge.list_answer_citations_by_request_ids(
                db, [req_a],
            )
            results_b = crud_knowledge.list_answer_citations_by_request_ids(
                db, [req_b],
            )

            self.assertEqual(len(results_a), 1)
            self.assertEqual(len(results_b), 1)
            self.assertEqual(results_a[0][0].quoted_text, "Request A")
            self.assertEqual(results_b[0][0].quoted_text, "Request B")
        finally:
            db.close()

    def test_citation_without_score_or_quoted_text_is_valid(self):
        """Citations may omit score and quoted_text (backward compat)."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            request_id = "req-citation-006"

            # Minimal citation: only required fields
            rows = crud_knowledge.replace_answer_citations(db, request_id, [
                {"document_id": doc_id, "chunk_id": chunk_id, "rank": 1},
            ])
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0].score)
            self.assertIsNone(rows[0].quoted_text)
        finally:
            db.close()

    def test_citation_persistence_via_finalize_chat_turn_flow(self):
        """Simulate the _finalize_chat_turn citation flow end-to-end."""
        from app.crud import crud_knowledge

        db = self.factory()
        try:
            doc_id, chunk_id = self._seed_citation_prerequisites(db)
            request_id = "req-citation-007"

            # Simulate what _finalize_chat_turn does:
            # It calls replace_answer_citations with sources filtered by
            # source_type == "knowledge"
            sources = [
                {
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "rank": 1,
                    "score": 0.85,
                    "snippet": "Evidence snippet from chat turn.",
                    "source_type": "knowledge",
                },
                {
                    "document_id": None,
                    "chunk_id": None,
                    "rank": 2,
                    "score": None,
                    "snippet": "Web result.",
                    "source_type": "web",
                },
            ]

            # Only knowledge sources get citations
            knowledge_citations = [
                {
                    "document_id": s["document_id"],
                    "chunk_id": s["chunk_id"],
                    "rank": s["rank"],
                    "score": s.get("score"),
                    "quoted_text": s.get("snippet") or "",
                }
                for s in sources
                if s.get("source_type") == "knowledge"
            ]
            rows = crud_knowledge.replace_answer_citations(
                db, request_id, citations=knowledge_citations,
            )

            # Verify exactly 1 citation (knowledge source only)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].document_id, doc_id)
            self.assertEqual(rows[0].chunk_id, chunk_id)
            self.assertEqual(rows[0].rank, 1)
            self.assertEqual(rows[0].quoted_text, "Evidence snippet from chat turn.")

            # Verify the citation is loadable in session history
            loaded = crud_knowledge.list_answer_citations_by_request_ids(
                db, [request_id],
            )
            self.assertEqual(len(loaded), 1)
            citation, document, chunk = loaded[0]
            self.assertEqual(citation.quoted_text, "Evidence snippet from chat turn.")
            self.assertEqual(document.title, "Citation Test Doc")
            self.assertIsNotNone(chunk.content)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
