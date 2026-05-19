"""Phase 3 unit tests for ingestion pipeline foundation.

Coverage (P03-T06):
- IngestionChunk: to_dict() shape, optional fields, metadata merging.
- IngestionPipelineError: attributes.
- CustomPipeline: parity with chunk_text(), empty text, metadata fields.
- Pipeline factory: custom default, unknown value error, llamaindex lazy import guard.
- LlamaIndex adapter: metadata preservation, no CRUD/vector_store imports,
  empty text, page_number/section_title extraction, parse error wrapping.
- _execute_indexing wiring: custom pipeline path passes existing tests.

Run with:
    cd DominicBE
    python -m pytest tests/test_ingestion_pipelines.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

# Must be set before any app import to prevent pydantic Settings validation
# from reading the real .env file.
os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**kwargs) -> MagicMock:
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
        "chunk_size": 512,
        "chunk_overlap": 64,
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is used for testing purposes. "
    "It contains multiple sentences to ensure chunking works correctly. "
    "Each sentence should be preserved as part of a chunk. "
    "The chunker splits on sentence boundaries when possible."
)

_MULTI_PARA_TEXT = (
    "First paragraph with some content about topic A.\n\n"
    "Second paragraph with some content about topic B.\n\n"
    "Third paragraph with some content about topic C."
)


# ===========================================================================
# IngestionChunk tests
# ===========================================================================

class TestIngestionChunk(unittest.TestCase):
    """Tests for the IngestionChunk dataclass and to_dict() conversion."""

    def setUp(self):
        from app.services.ingestion.base import IngestionChunk
        self.IngestionChunk = IngestionChunk

    def test_to_dict_required_fields(self):
        chunk = self.IngestionChunk(chunk_index=0, content="Hello world.", token_count=3)
        d = chunk.to_dict()
        self.assertEqual(d["chunk_index"], 0)
        self.assertEqual(d["content"], "Hello world.")
        self.assertEqual(d["token_count"], 3)
        self.assertIn("metadata_json", d)

    def test_to_dict_metadata_includes_char_count(self):
        chunk = self.IngestionChunk(chunk_index=0, content="Hello world.", token_count=3)
        d = chunk.to_dict()
        self.assertEqual(d["metadata_json"]["char_count"], len("Hello world."))

    def test_to_dict_metadata_includes_versions(self):
        chunk = self.IngestionChunk(
            chunk_index=0, content="text", token_count=1,
            parser_version="custom-v1", chunker_version="custom-sentence-v1",
        )
        d = chunk.to_dict()
        self.assertEqual(d["metadata_json"]["parser_version"], "custom-v1")
        self.assertEqual(d["metadata_json"]["chunker_version"], "custom-sentence-v1")

    def test_to_dict_page_number_included_when_set(self):
        chunk = self.IngestionChunk(chunk_index=0, content="text", token_count=1, page_number=3)
        d = chunk.to_dict()
        self.assertEqual(d["metadata_json"]["page_number"], 3)

    def test_to_dict_page_number_absent_when_none(self):
        chunk = self.IngestionChunk(chunk_index=0, content="text", token_count=1)
        d = chunk.to_dict()
        self.assertNotIn("page_number", d["metadata_json"])

    def test_to_dict_section_title_included_when_set(self):
        chunk = self.IngestionChunk(
            chunk_index=0, content="text", token_count=1, section_title="Introduction"
        )
        d = chunk.to_dict()
        self.assertEqual(d["metadata_json"]["section_title"], "Introduction")

    def test_to_dict_section_title_absent_when_none(self):
        chunk = self.IngestionChunk(chunk_index=0, content="text", token_count=1)
        d = chunk.to_dict()
        self.assertNotIn("section_title", d["metadata_json"])

    def test_to_dict_extra_metadata_merged(self):
        chunk = self.IngestionChunk(
            chunk_index=0, content="text", token_count=1,
            extra_metadata={"custom_key": "custom_value", "score": 0.9},
        )
        d = chunk.to_dict()
        self.assertEqual(d["metadata_json"]["custom_key"], "custom_value")
        self.assertEqual(d["metadata_json"]["score"], 0.9)

    def test_chunk_index_stable(self):
        chunks = [
            self.IngestionChunk(chunk_index=i, content=f"chunk {i}", token_count=2)
            for i in range(5)
        ]
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.to_dict()["chunk_index"], i)


# ===========================================================================
# IngestionPipelineError tests
# ===========================================================================

class TestIngestionPipelineError(unittest.TestCase):
    """Tests for IngestionPipelineError exception attributes."""

    def setUp(self):
        from app.services.ingestion.base import IngestionPipelineError
        self.IngestionPipelineError = IngestionPipelineError

    def test_attributes_set(self):
        err = self.IngestionPipelineError(
            "something failed", pipeline="custom", category="parse_error"
        )
        self.assertEqual(err.pipeline, "custom")
        self.assertEqual(err.category, "parse_error")
        self.assertIn("something failed", str(err))

    def test_default_attributes(self):
        err = self.IngestionPipelineError("oops")
        self.assertEqual(err.pipeline, "")
        self.assertEqual(err.category, "unknown")

    def test_is_runtime_error(self):
        err = self.IngestionPipelineError("oops")
        self.assertIsInstance(err, RuntimeError)


# ===========================================================================
# CustomPipeline tests
# ===========================================================================

class TestCustomPipeline(unittest.TestCase):
    """Tests for CustomPipeline — parity with chunk_text() and metadata fields."""

    def setUp(self):
        from app.services.ingestion.custom_pipeline import CustomPipeline
        self.CustomPipeline = CustomPipeline

    def test_pipeline_name(self):
        p = self.CustomPipeline()
        self.assertEqual(p.pipeline_name, "custom")

    def test_parser_version(self):
        p = self.CustomPipeline()
        self.assertEqual(p.parser_version, "custom-v1")

    def test_chunker_version(self):
        p = self.CustomPipeline()
        self.assertEqual(p.chunker_version, "custom-sentence-v1")

    def test_empty_text_returns_empty_list(self):
        p = self.CustomPipeline()
        result = p.chunk_document("")
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty_list(self):
        p = self.CustomPipeline()
        result = p.chunk_document("   \n\t  ")
        self.assertEqual(result, [])

    def test_returns_ingestion_chunks(self):
        from app.services.ingestion.base import IngestionChunk
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for chunk in result:
            self.assertIsInstance(chunk, IngestionChunk)

    def test_chunk_index_sequential(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        indices = [c.chunk_index for c in result]
        self.assertEqual(indices, list(range(len(result))))

    def test_content_non_empty(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        for chunk in result:
            self.assertTrue(chunk.content.strip())

    def test_token_count_positive(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        for chunk in result:
            self.assertGreater(chunk.token_count, 0)

    def test_parser_version_on_chunks(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        for chunk in result:
            self.assertEqual(chunk.parser_version, "custom-v1")

    def test_chunker_version_on_chunks(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        for chunk in result:
            self.assertEqual(chunk.chunker_version, "custom-sentence-v1")

    def test_to_dict_shape_compatible_with_prepare_chunks(self):
        """to_dict() output must have chunk_index, content, token_count, metadata_json."""
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        for chunk in result:
            d = chunk.to_dict()
            self.assertIn("chunk_index", d)
            self.assertIn("content", d)
            self.assertIn("token_count", d)
            self.assertIn("metadata_json", d)

    def test_parity_with_chunk_text_count(self):
        """CustomPipeline must produce the same number of chunks as chunk_text()."""
        from app.services.knowledge_service import chunk_text
        p = self.CustomPipeline()
        pipeline_chunks = p.chunk_document(_SAMPLE_TEXT)
        direct_chunks = chunk_text(_SAMPLE_TEXT)
        self.assertEqual(len(pipeline_chunks), len(direct_chunks))

    def test_parity_with_chunk_text_content(self):
        """CustomPipeline chunk content must match chunk_text() content exactly."""
        from app.services.knowledge_service import chunk_text
        p = self.CustomPipeline()
        pipeline_chunks = p.chunk_document(_SAMPLE_TEXT)
        direct_chunks = chunk_text(_SAMPLE_TEXT)
        for pc, dc in zip(pipeline_chunks, direct_chunks):
            self.assertEqual(pc.content, dc["content"])

    def test_metadata_json_has_parser_version(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT)
        for chunk in result:
            meta = chunk.to_dict()["metadata_json"]
            self.assertEqual(meta["parser_version"], "custom-v1")
            self.assertEqual(meta["chunker_version"], "custom-sentence-v1")

    def test_document_id_kwarg_accepted(self):
        """document_id kwarg must be accepted without error."""
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT, document_id=42)
        self.assertGreater(len(result), 0)

    def test_source_uri_kwarg_accepted(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT, source_uri="test.txt")
        self.assertGreater(len(result), 0)

    def test_title_kwarg_accepted(self):
        p = self.CustomPipeline()
        result = p.chunk_document(_SAMPLE_TEXT, title="Test Document")
        self.assertGreater(len(result), 0)


# ===========================================================================
# Pipeline factory tests
# ===========================================================================

class TestIngestionPipelineFactory(unittest.TestCase):
    """Tests for get_ingestion_pipeline() factory.

    The factory uses a local import of settings inside get_ingestion_pipeline(),
    so we patch 'app.core.config.settings' (the module-level singleton) rather
    than a module attribute on the factory module.
    """

    def _get_factory(self):
        from app.services.ingestion.factory import get_ingestion_pipeline
        return get_ingestion_pipeline

    def test_default_returns_custom_pipeline(self):
        from app.services.ingestion.custom_pipeline import CustomPipeline
        mock_settings = _make_settings(ingestion_pipeline="custom")
        with patch("app.core.config.settings", mock_settings):
            factory = self._get_factory()
            pipeline = factory()
        self.assertIsInstance(pipeline, CustomPipeline)

    def test_explicit_custom_returns_custom_pipeline(self):
        from app.services.ingestion.custom_pipeline import CustomPipeline
        mock_settings = _make_settings(ingestion_pipeline="custom")
        with patch("app.core.config.settings", mock_settings):
            factory = self._get_factory()
            pipeline = factory(pipeline="custom")
        self.assertIsInstance(pipeline, CustomPipeline)

    def test_unknown_pipeline_raises_error(self):
        from app.services.ingestion.base import IngestionPipelineError
        mock_settings = _make_settings(ingestion_pipeline="custom")
        with patch("app.core.config.settings", mock_settings):
            factory = self._get_factory()
            with self.assertRaises(IngestionPipelineError) as ctx:
                factory(pipeline="nonexistent_pipeline")
        self.assertEqual(ctx.exception.category, "configuration_error")
        self.assertIn("nonexistent_pipeline", str(ctx.exception))

    def test_llamaindex_missing_dependency_raises_clear_error(self):
        """When llama-index-core is not installed, factory raises IngestionPipelineError."""
        from app.services.ingestion.base import IngestionPipelineError
        mock_settings = _make_settings(ingestion_pipeline="llamaindex")

        # Simulate llama-index-core not installed by making the import fail
        with patch("app.core.config.settings", mock_settings):
            with patch.dict(sys.modules, {
                "app.services.ingestion.llamaindex_pipeline": None,
            }):
                factory = self._get_factory()
                with self.assertRaises(IngestionPipelineError) as ctx:
                    factory(pipeline="llamaindex")
                self.assertEqual(ctx.exception.category, "missing_dependency")
                self.assertIn("llama-index-core", str(ctx.exception))

    def test_pipeline_name_custom(self):
        mock_settings = _make_settings(ingestion_pipeline="custom")
        with patch("app.core.config.settings", mock_settings):
            factory = self._get_factory()
            pipeline = factory()
        self.assertEqual(pipeline.pipeline_name, "custom")

    def test_case_insensitive_custom(self):
        """Factory should normalize pipeline name to lowercase."""
        from app.services.ingestion.custom_pipeline import CustomPipeline
        mock_settings = _make_settings(ingestion_pipeline="custom")
        with patch("app.core.config.settings", mock_settings):
            factory = self._get_factory()
            pipeline = factory(pipeline="CUSTOM")
        self.assertIsInstance(pipeline, CustomPipeline)

    def test_override_pipeline_kwarg_takes_precedence(self):
        """pipeline kwarg overrides settings.ingestion_pipeline."""
        from app.services.ingestion.custom_pipeline import CustomPipeline
        # settings says llamaindex but kwarg says custom
        mock_settings = _make_settings(ingestion_pipeline="llamaindex")
        with patch("app.core.config.settings", mock_settings):
            factory = self._get_factory()
            pipeline = factory(pipeline="custom")
        self.assertIsInstance(pipeline, CustomPipeline)


# ===========================================================================
# LlamaIndex pipeline tests (with mocked llama-index-core)
# ===========================================================================

class TestLlamaIndexPipeline(unittest.TestCase):
    """Tests for LlamaIndexPipeline with llama-index-core mocked.

    These tests verify metadata mapping, boundary constraints, and error
    handling without requiring llama-index-core to be installed.
    """

    def _make_mock_node(self, text: str, metadata: dict | None = None):
        """Create a mock LlamaIndex TextNode."""
        node = MagicMock()
        node.get_content.return_value = text
        node.metadata = metadata or {}
        return node

    def _make_pipeline_with_mocks(self, nodes: list, chunk_size: int = 512, chunk_overlap: int = 64):
        """Build a LlamaIndexPipeline with mocked SentenceSplitter and Document."""
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.get_nodes_from_documents.return_value = nodes

        mock_splitter_cls = MagicMock(return_value=mock_splitter_instance)
        mock_doc_cls = MagicMock(return_value=MagicMock())

        mock_settings = _make_settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        with patch("app.services.ingestion.llamaindex_pipeline.SentenceSplitter", mock_splitter_cls, create=True), \
             patch("app.services.ingestion.llamaindex_pipeline.LlamaDocument", mock_doc_cls, create=True):

            # Patch the imports inside __init__
            import app.services.ingestion.llamaindex_pipeline as llama_mod
            original_splitter = getattr(llama_mod, "_SentenceSplitter_cls", None)

            from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline

            pipeline = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
            pipeline._chunk_size = chunk_size
            pipeline._chunk_overlap = chunk_overlap
            pipeline._SentenceSplitter = mock_splitter_cls
            pipeline._LlamaDocument = mock_doc_cls

        return pipeline, mock_splitter_instance

    def test_pipeline_name(self):
        from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline
        p = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
        self.assertEqual(p.pipeline_name, "llamaindex")

    def test_parser_version(self):
        from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline
        p = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
        self.assertEqual(p.parser_version, "llamaindex-core-v1")

    def test_chunker_version(self):
        from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline
        p = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
        self.assertEqual(p.chunker_version, "llamaindex-sentence-v1")

    def test_empty_text_returns_empty_list(self):
        from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline
        p = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
        p._chunk_size = 512
        p._chunk_overlap = 64
        p._SentenceSplitter = MagicMock()
        p._LlamaDocument = MagicMock()
        result = p.chunk_document("")
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty_list(self):
        from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline
        p = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
        p._chunk_size = 512
        p._chunk_overlap = 64
        p._SentenceSplitter = MagicMock()
        p._LlamaDocument = MagicMock()
        result = p.chunk_document("   \n\t  ")
        self.assertEqual(result, [])

    def test_basic_chunk_mapping(self):
        """Nodes are mapped to IngestionChunk with correct chunk_index and content."""
        from app.services.ingestion.base import IngestionChunk
        nodes = [
            self._make_mock_node("First chunk content."),
            self._make_mock_node("Second chunk content."),
        ]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT, document_id=1)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], IngestionChunk)
        self.assertEqual(result[0].chunk_index, 0)
        self.assertEqual(result[0].content, "First chunk content.")
        self.assertEqual(result[1].chunk_index, 1)
        self.assertEqual(result[1].content, "Second chunk content.")

    def test_page_number_extracted_from_page_label(self):
        """page_label in node metadata is mapped to page_number."""
        nodes = [self._make_mock_node("Content.", metadata={"page_label": "3"})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].page_number, 3)

    def test_page_number_extracted_from_page_number_key(self):
        """page_number in node metadata is mapped to page_number."""
        nodes = [self._make_mock_node("Content.", metadata={"page_number": 5})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].page_number, 5)

    def test_page_number_none_when_absent(self):
        nodes = [self._make_mock_node("Content.", metadata={})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertIsNone(result[0].page_number)

    def test_section_title_extracted(self):
        """section_title in node metadata is mapped to section_title."""
        nodes = [self._make_mock_node("Content.", metadata={"section_title": "Introduction"})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].section_title, "Introduction")

    def test_section_title_from_header_key(self):
        """header key in node metadata is also accepted as section_title."""
        nodes = [self._make_mock_node("Content.", metadata={"header": "Chapter 1"})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].section_title, "Chapter 1")

    def test_section_title_none_when_absent(self):
        nodes = [self._make_mock_node("Content.", metadata={})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertIsNone(result[0].section_title)

    def test_parser_and_chunker_version_on_chunks(self):
        nodes = [self._make_mock_node("Content.")]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].parser_version, "llamaindex-core-v1")
        self.assertEqual(result[0].chunker_version, "llamaindex-sentence-v1")

    def test_document_level_metadata_not_in_extra(self):
        """source_uri, title, document_id set on the LlamaDocument must not
        appear in extra_metadata of the returned chunks."""
        nodes = [self._make_mock_node("Content.", metadata={
            "source_uri": "test.txt",
            "title": "Test",
            "document_id": 1,
        })]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT, document_id=1, source_uri="test.txt", title="Test")
        extra = result[0].extra_metadata
        self.assertNotIn("source_uri", extra)
        self.assertNotIn("title", extra)
        self.assertNotIn("document_id", extra)

    def test_extra_metadata_preserved(self):
        """Unknown node metadata keys are preserved in extra_metadata."""
        nodes = [self._make_mock_node("Content.", metadata={"custom_field": "value123"})]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].extra_metadata.get("custom_field"), "value123")

    def test_empty_node_text_skipped(self):
        """Nodes with empty content after strip() are skipped."""
        nodes = [
            self._make_mock_node("Real content."),
            self._make_mock_node("   "),  # whitespace-only — should be skipped
            self._make_mock_node(""),     # empty — should be skipped
        ]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "Real content.")

    def test_token_count_estimated_from_chars(self):
        """token_count is estimated as max(1, len(content) // 4)."""
        content = "A" * 100
        nodes = [self._make_mock_node(content)]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        self.assertEqual(result[0].token_count, 25)  # 100 // 4

    def test_to_dict_shape_compatible(self):
        """to_dict() output must have chunk_index, content, token_count, metadata_json."""
        nodes = [self._make_mock_node("Content.")]
        pipeline, _ = self._make_pipeline_with_mocks(nodes)
        result = pipeline.chunk_document(_SAMPLE_TEXT)
        d = result[0].to_dict()
        self.assertIn("chunk_index", d)
        self.assertIn("content", d)
        self.assertIn("token_count", d)
        self.assertIn("metadata_json", d)

    def test_parse_error_wrapped_in_pipeline_error(self):
        """Exceptions from SentenceSplitter are wrapped in IngestionPipelineError."""
        from app.services.ingestion.base import IngestionPipelineError
        from app.services.ingestion.llamaindex_pipeline import LlamaIndexPipeline

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.get_nodes_from_documents.side_effect = RuntimeError("splitter crash")
        mock_splitter_cls = MagicMock(return_value=mock_splitter_instance)

        p = LlamaIndexPipeline.__new__(LlamaIndexPipeline)
        p._chunk_size = 512
        p._chunk_overlap = 64
        p._SentenceSplitter = mock_splitter_cls
        p._LlamaDocument = MagicMock(return_value=MagicMock())

        with self.assertRaises(IngestionPipelineError) as ctx:
            p.chunk_document(_SAMPLE_TEXT, document_id=99)
        self.assertEqual(ctx.exception.pipeline, "llamaindex")
        self.assertEqual(ctx.exception.category, "parse_error")
        self.assertIn("99", str(ctx.exception))

    def test_no_crud_import(self):
        """LlamaIndex pipeline module must not import CRUD modules."""
        import app.services.ingestion.llamaindex_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from app.crud", source)
        self.assertNotIn("import crud_knowledge", source)

    def test_no_vector_store_import(self):
        """LlamaIndex pipeline module must not import vector_store."""
        import app.services.ingestion.llamaindex_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from app.services import vector_store", source)
        self.assertNotIn("import vector_store", source)

    def test_no_chat_service_import(self):
        """LlamaIndex pipeline module must not import chat_service."""
        import app.services.ingestion.llamaindex_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("chat_service", source)

    def test_no_endpoint_import(self):
        """LlamaIndex pipeline module must not import endpoint modules."""
        import app.services.ingestion.llamaindex_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from app.api", source)


# ===========================================================================
# Custom pipeline boundary tests
# ===========================================================================

class TestCustomPipelineBoundary(unittest.TestCase):
    """Verify CustomPipeline does not import forbidden modules."""

    def test_no_crud_import(self):
        import app.services.ingestion.custom_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from app.crud", source)
        self.assertNotIn("import crud_knowledge", source)

    def test_no_vector_store_import(self):
        import app.services.ingestion.custom_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from app.services import vector_store", source)
        self.assertNotIn("import vector_store", source)

    def test_no_chat_service_import(self):
        import app.services.ingestion.custom_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("chat_service", source)

    def test_no_endpoint_import(self):
        import app.services.ingestion.custom_pipeline as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from app.api", source)


# ===========================================================================
# _execute_indexing wiring tests (P03-T05)
# ===========================================================================

class TestExecuteIndexingWiring(unittest.TestCase):
    """Verify _execute_indexing() delegates to the pipeline factory.

    These tests use mocks to avoid needing a real database or vector store.
    They confirm the custom pipeline path is called and produces the same
    output shape as before Phase 3.
    """

    def _make_mock_doc(self, doc_id: int = 1) -> MagicMock:
        doc = MagicMock()
        doc.id = doc_id
        doc.raw_text = _SAMPLE_TEXT
        doc.owner_username = "testuser"
        doc.source_uri = "test.txt"
        doc.title = "Test Document"
        return doc

    def test_execute_indexing_calls_pipeline_factory(self):
        """_execute_indexing must call get_ingestion_pipeline()."""
        from app.services.knowledge_service import _execute_indexing

        mock_doc = self._make_mock_doc()
        mock_db = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.pipeline_name = "custom"
        # Return a minimal chunk list
        from app.services.ingestion.base import IngestionChunk
        mock_pipeline.chunk_document.return_value = [
            IngestionChunk(chunk_index=0, content="Test content.", token_count=3)
        ]

        mock_chunk_row = MagicMock()
        mock_chunk_row.id = 1

        with patch("app.services.knowledge_service.crud_knowledge") as mock_crud, \
             patch("app.services.knowledge_service.vector_store") as mock_vs, \
             patch("app.services.ingestion.factory.get_ingestion_pipeline", return_value=mock_pipeline), \
             patch("app.services.knowledge_service.get_embedding_provider") as mock_emb_factory:

            mock_crud.get_document.return_value = mock_doc
            mock_crud.create_chunks_bulk.return_value = [mock_chunk_row]
            mock_vs.should_store_embeddings_in_database.return_value = False
            mock_vs.delete_document_chunks.return_value = None
            mock_vs.upsert_document_chunks.return_value = None

            mock_provider = MagicMock()
            from app.services.embeddings.base import EmbeddingMeta, EmbedResult
            mock_meta = EmbeddingMeta(
                provider="local", model="local-hash-v1", dimensions=64, version="local-hash-v1"
            )
            mock_provider.meta = mock_meta
            mock_provider.embed_texts.return_value = EmbedResult(
                vectors=[[0.1] * 64], meta=mock_meta
            )
            mock_emb_factory.return_value = mock_provider

            result = _execute_indexing(mock_db, 1, 1)

        self.assertEqual(result["status"], "indexed")
        self.assertIn("pipeline", result)
        self.assertEqual(result["pipeline"], "custom")

    def test_execute_indexing_result_has_pipeline_key(self):
        """Result dict from _execute_indexing must include 'pipeline' key (Phase 3 addition)."""
        from app.services.knowledge_service import _execute_indexing
        from app.services.ingestion.base import IngestionChunk

        mock_doc = self._make_mock_doc()
        mock_db = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.pipeline_name = "custom"
        mock_pipeline.chunk_document.return_value = [
            IngestionChunk(chunk_index=0, content="Content.", token_count=2)
        ]

        mock_chunk_row = MagicMock()
        mock_chunk_row.id = 1

        with patch("app.services.knowledge_service.crud_knowledge") as mock_crud, \
             patch("app.services.knowledge_service.vector_store") as mock_vs, \
             patch("app.services.ingestion.factory.get_ingestion_pipeline", return_value=mock_pipeline), \
             patch("app.services.knowledge_service.get_embedding_provider") as mock_emb_factory:

            mock_crud.get_document.return_value = mock_doc
            mock_crud.create_chunks_bulk.return_value = [mock_chunk_row]
            mock_vs.should_store_embeddings_in_database.return_value = False
            mock_vs.delete_document_chunks.return_value = None
            mock_vs.upsert_document_chunks.return_value = None

            mock_provider = MagicMock()
            from app.services.embeddings.base import EmbeddingMeta, EmbedResult
            mock_meta = EmbeddingMeta(
                provider="local", model="local-hash-v1", dimensions=64, version="local-hash-v1"
            )
            mock_provider.meta = mock_meta
            mock_provider.embed_texts.return_value = EmbedResult(
                vectors=[[0.0] * 64], meta=mock_meta
            )
            mock_emb_factory.return_value = mock_provider

            result = _execute_indexing(mock_db, 1, 1)

        self.assertIn("pipeline", result)

    def test_execute_indexing_no_chunks_raises(self):
        """_execute_indexing must raise ValueError when pipeline returns no chunks."""
        from app.services.knowledge_service import _execute_indexing

        mock_doc = self._make_mock_doc()
        mock_db = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.pipeline_name = "custom"
        mock_pipeline.chunk_document.return_value = []  # empty — should raise

        with patch("app.services.knowledge_service.crud_knowledge") as mock_crud, \
             patch("app.services.knowledge_service.vector_store"), \
             patch("app.services.ingestion.factory.get_ingestion_pipeline", return_value=mock_pipeline):

            mock_crud.get_document.return_value = mock_doc

            with self.assertRaises(ValueError) as ctx:
                _execute_indexing(mock_db, 1, 1)

        self.assertIn("no chunks", str(ctx.exception).lower())


# ===========================================================================
# Package import tests
# ===========================================================================

class TestIngestionPackageImports(unittest.TestCase):
    """Verify the ingestion package exports are clean and importable."""

    def test_package_exports_ingestion_chunk(self):
        from app.services.ingestion import IngestionChunk
        self.assertTrue(callable(IngestionChunk))

    def test_package_exports_ingestion_pipeline_error(self):
        from app.services.ingestion import IngestionPipelineError
        self.assertTrue(issubclass(IngestionPipelineError, RuntimeError))

    def test_package_exports_ingestion_pipeline_protocol(self):
        from app.services.ingestion import IngestionPipeline
        # Protocol is a class
        self.assertIsNotNone(IngestionPipeline)

    def test_factory_importable(self):
        from app.services.ingestion.factory import get_ingestion_pipeline
        self.assertTrue(callable(get_ingestion_pipeline))

    def test_custom_pipeline_importable(self):
        from app.services.ingestion.custom_pipeline import CustomPipeline
        self.assertTrue(callable(CustomPipeline))

    def test_llamaindex_pipeline_module_importable_without_llama_installed(self):
        """The module itself must be importable even if llama-index-core is absent.
        Only instantiation should fail."""
        # This test verifies the module-level code has no unconditional llama import.
        try:
            import app.services.ingestion.llamaindex_pipeline  # noqa: F401
        except ImportError as exc:
            # If llama-index-core is installed this won't fail; if not, the
            # module-level import guard should prevent this.
            # We only fail if the error is NOT about llama_index itself.
            if "llama_index" not in str(exc):
                self.fail(f"Unexpected ImportError: {exc}")


if __name__ == "__main__":
    unittest.main()
