"""Phase 1 unit tests for embedding provider foundation.

Coverage:
- Local hash provider: deterministic output, dimension compatibility, empty text.
- Factory: local selection, ollama selection, unknown provider failure.
- Ollama provider: unavailable (connection error), timeout, invalid response shapes.
- Default retrieval path: unchanged local behavior via compute_text_embedding shim.

Run with:
    cd DominicBE
    python -m pytest tests/test_embedding_providers.py -v
"""
from __future__ import annotations

import math
import os
import unittest
from unittest.mock import MagicMock, patch

# Must be set before any app import to prevent pydantic Settings validation
# from reading the real .env file (which may have DEBUG=release or other
# non-boolean values that fail validation).
# Use hard assignment (not setdefault) so it overrides any existing env value.
os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"
# Do NOT set DATABASE_URL to sqlite – SQLite doesn't support pool_timeout/max_overflow.
# The embedding provider tests do not need a real database connection.

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
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ===========================================================================
# Local hash provider tests
# ===========================================================================

class TestLocalHashProvider(unittest.TestCase):
    """Tests for LocalHashProvider – backward compatibility with compute_text_embedding."""

    def _make_provider(self, dimensions: int = 64):
        from app.services.embeddings.local_hash_provider import LocalHashProvider
        return LocalHashProvider(model="local-hash-v1", dimensions=dimensions)

    def test_embed_texts_returns_correct_count(self):
        provider = self._make_provider()
        result = provider.embed_texts(["hello world", "foo bar"])
        self.assertEqual(len(result.vectors), 2)

    def test_embed_texts_dimension_matches(self):
        provider = self._make_provider(dimensions=64)
        result = provider.embed_texts(["test text"])
        self.assertEqual(len(result.vectors[0]), 64)

    def test_embed_texts_deterministic(self):
        """Same input must always produce the same vector."""
        provider = self._make_provider()
        r1 = provider.embed_texts(["deterministic test"])
        r2 = provider.embed_texts(["deterministic test"])
        self.assertEqual(r1.vectors[0], r2.vectors[0])

    def test_embed_texts_different_inputs_differ(self):
        provider = self._make_provider()
        r1 = provider.embed_texts(["hello world"])
        r2 = provider.embed_texts(["completely different text"])
        self.assertNotEqual(r1.vectors[0], r2.vectors[0])

    def test_embed_texts_normalized_vector(self):
        """Non-empty text should produce a unit-normalized vector (magnitude ≈ 1)."""
        provider = self._make_provider()
        result = provider.embed_texts(["normalization test"])
        vec = result.vectors[0]
        magnitude = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(magnitude, 1.0, places=4)

    def test_embed_texts_empty_text_returns_zero_vector(self):
        provider = self._make_provider()
        result = provider.embed_texts([""])
        self.assertEqual(result.vectors[0], [0.0] * 64)

    def test_embed_texts_empty_list_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError
        provider = self._make_provider()
        with self.assertRaises(EmbeddingProviderError):
            provider.embed_texts([])

    def test_embed_query_returns_single_vector(self):
        provider = self._make_provider()
        result = provider.embed_query("what is the refund policy?")
        self.assertEqual(len(result.vector), 64)

    def test_embed_query_empty_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError
        provider = self._make_provider()
        with self.assertRaises(EmbeddingProviderError):
            provider.embed_query("")

    def test_embed_query_matches_embed_texts(self):
        """embed_query and embed_texts must produce the same vector for the same text."""
        provider = self._make_provider()
        text = "consistent embedding test"
        query_result = provider.embed_query(text)
        batch_result = provider.embed_texts([text])
        self.assertEqual(query_result.vector, batch_result.vectors[0])

    def test_meta_fields(self):
        provider = self._make_provider()
        meta = provider.meta
        self.assertEqual(meta.provider, "local")
        self.assertEqual(meta.model, "local-hash-v1")
        self.assertEqual(meta.dimensions, 64)
        self.assertEqual(meta.version, "local-hash-v1")

    def test_backward_compat_with_compute_text_embedding(self):
        """LocalHashProvider must produce the same output as the original compute_text_embedding."""
        from app.services.embeddings.local_hash_provider import _hash_embed
        from app.services.knowledge_service import compute_text_embedding

        text = "backward compatibility check"
        original = compute_text_embedding(text, dimensions=64)
        provider_vec = _hash_embed(text, 64)
        self.assertEqual(original, provider_vec)

    def test_custom_dimensions(self):
        provider = self._make_provider(dimensions=128)
        result = provider.embed_texts(["dimension test"])
        self.assertEqual(len(result.vectors[0]), 128)


# ===========================================================================
# Factory tests
# ===========================================================================

class TestEmbeddingProviderFactory(unittest.TestCase):
    """Tests for get_embedding_provider factory."""

    def test_local_provider_returned_by_default(self):
        from app.services.embeddings.factory import get_embedding_provider
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        with patch("app.services.embeddings.factory.settings", _make_settings_override(embedding_provider="local")):
            provider = get_embedding_provider()
        self.assertIsInstance(provider, LocalHashProvider)

    def test_ollama_provider_returned_when_configured(self):
        from app.services.embeddings.factory import get_embedding_provider
        from app.services.embeddings.ollama_provider import OllamaProvider

        with patch("app.services.embeddings.factory.settings", _make_settings_override(embedding_provider="ollama")):
            provider = get_embedding_provider()
        self.assertIsInstance(provider, OllamaProvider)

    def test_unknown_provider_raises_clear_error(self):
        from app.services.embeddings.base import EmbeddingProviderError
        from app.services.embeddings.factory import get_embedding_provider

        with patch("app.services.embeddings.factory.settings", _make_settings_override(embedding_provider="faiss")):
            with self.assertRaises(EmbeddingProviderError) as ctx:
                get_embedding_provider()
        self.assertIn("faiss", str(ctx.exception).lower())
        self.assertIn("configuration_error", ctx.exception.category)

    def test_factory_override_provider_name(self):
        from app.services.embeddings.factory import get_embedding_provider
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        # Even if settings say ollama, explicit override wins
        with patch("app.services.embeddings.factory.settings", _make_settings_override(embedding_provider="ollama")):
            provider = get_embedding_provider(provider_name="local")
        self.assertIsInstance(provider, LocalHashProvider)

    def test_factory_override_model(self):
        from app.services.embeddings.factory import get_embedding_provider

        with patch("app.services.embeddings.factory.settings", _make_settings_override()):
            provider = get_embedding_provider(model="custom-model-v2")
        self.assertEqual(provider.meta.model, "custom-model-v2")

    def test_factory_override_dimensions(self):
        from app.services.embeddings.factory import get_embedding_provider

        with patch("app.services.embeddings.factory.settings", _make_settings_override()):
            provider = get_embedding_provider(dimensions=128)
        self.assertEqual(provider.meta.dimensions, 128)

    def test_no_service_hard_codes_qwen_model(self):
        """Factory must not hard-code qwen3-embedding:0.6b; model comes from config."""
        from app.services.embeddings import factory as factory_module
        import inspect
        source = inspect.getsource(factory_module)
        self.assertNotIn("qwen3-embedding:0.6b", source)


# ===========================================================================
# Ollama provider tests (mocked HTTP)
# ===========================================================================

class TestOllamaProvider(unittest.TestCase):
    """Tests for OllamaProvider with mocked httpx calls."""

    def _make_provider(self, **kwargs):
        from app.services.embeddings.ollama_provider import OllamaProvider
        defaults = {
            "model": "test-model:latest",
            "base_url": "http://localhost:11434",
            "timeout_seconds": 5.0,
            "batch_size": 4,
            "expected_dimensions": 0,
        }
        defaults.update(kwargs)
        return OllamaProvider(**defaults)

    def _mock_response(self, embeddings: list[list[float]], status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.is_success = (status_code < 400)
        mock_resp.json.return_value = {"embeddings": embeddings}
        return mock_resp

    def test_embed_texts_happy_path(self):
        provider = self._make_provider()
        fake_vec = [0.1, 0.2, 0.3]
        mock_resp = self._mock_response([fake_vec, fake_vec])

        with patch("httpx.post", return_value=mock_resp):
            result = provider.embed_texts(["text one", "text two"])

        self.assertEqual(len(result.vectors), 2)
        self.assertEqual(result.vectors[0], [float(v) for v in fake_vec])
        self.assertEqual(result.meta.provider, "ollama")

    def test_embed_query_happy_path(self):
        provider = self._make_provider()
        fake_vec = [0.5, 0.6, 0.7]
        mock_resp = self._mock_response([fake_vec])

        with patch("httpx.post", return_value=mock_resp):
            result = provider.embed_query("what is the policy?")

        self.assertEqual(result.vector, [float(v) for v in fake_vec])
        self.assertEqual(result.meta.provider, "ollama")

    def test_connection_error_raises_provider_error(self):
        import httpx
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with self.assertRaises(EmbeddingProviderError) as ctx:
                provider.embed_texts(["test"])
        self.assertEqual(ctx.exception.category, "connection_error")
        self.assertEqual(ctx.exception.provider, "ollama")

    def test_timeout_raises_provider_error(self):
        import httpx
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with self.assertRaises(EmbeddingProviderError) as ctx:
                provider.embed_texts(["test"])
        self.assertEqual(ctx.exception.category, "timeout")

    def test_missing_embeddings_field_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {"model": "test"}  # no 'embeddings' key

        with patch("httpx.post", return_value=mock_resp):
            with self.assertRaises(EmbeddingProviderError) as ctx:
                provider.embed_texts(["test"])
        self.assertEqual(ctx.exception.category, "invalid_response")

    def test_count_mismatch_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        # Send 2 texts but response has only 1 embedding
        mock_resp = self._mock_response([[0.1, 0.2]])

        with patch("httpx.post", return_value=mock_resp):
            with self.assertRaises(EmbeddingProviderError) as ctx:
                provider.embed_texts(["text one", "text two"])
        self.assertEqual(ctx.exception.category, "count_mismatch")

    def test_non_numeric_values_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {"embeddings": [["nan_str", "inf_str"]]}

        with patch("httpx.post", return_value=mock_resp):
            with self.assertRaises(EmbeddingProviderError) as ctx:
                provider.embed_texts(["test"])
        self.assertEqual(ctx.exception.category, "non_numeric_values")

    def test_inconsistent_dimensions_raises(self):
        from app.services.embeddings.base import EmbeddingDimensionMismatchError

        provider = self._make_provider()
        # Two embeddings with different dimensions
        mock_resp = self._mock_response([[0.1, 0.2, 0.3], [0.4, 0.5]])

        with patch("httpx.post", return_value=mock_resp):
            with self.assertRaises(EmbeddingDimensionMismatchError):
                provider.embed_texts(["text one", "text two"])

    def test_http_500_retries_and_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.is_success = False

        with patch("httpx.post", return_value=mock_resp), \
             patch("time.sleep"):  # skip actual sleep in tests
            with self.assertRaises(EmbeddingProviderError) as ctx:
                provider.embed_texts(["test"])
        self.assertEqual(ctx.exception.category, "http_error")

    def test_http_400_does_not_retry(self):
        """HTTP 400 is not retryable – should raise immediately."""
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.is_success = False

        call_count = 0

        def counting_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch("httpx.post", side_effect=counting_post):
            with self.assertRaises(EmbeddingProviderError):
                provider.embed_texts(["test"])

        # Should only be called once (no retry for 400)
        self.assertEqual(call_count, 1)

    def test_empty_texts_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        with self.assertRaises(EmbeddingProviderError) as ctx:
            provider.embed_texts([])
        self.assertEqual(ctx.exception.category, "invalid_input")

    def test_empty_query_raises(self):
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        with self.assertRaises(EmbeddingProviderError) as ctx:
            provider.embed_query("")
        self.assertEqual(ctx.exception.category, "invalid_input")

    def test_no_raw_text_in_error_message(self):
        """Error messages must not include raw document text."""
        import httpx
        from app.services.embeddings.base import EmbeddingProviderError

        provider = self._make_provider()
        secret_text = "CONFIDENTIAL: do not log this text"

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            try:
                provider.embed_texts([secret_text])
            except EmbeddingProviderError as exc:
                self.assertNotIn(secret_text, str(exc))

    def test_batch_splits_correctly(self):
        """Provider must split large input into sub-batches of batch_size."""
        provider = self._make_provider(batch_size=2)
        fake_vec = [0.1, 0.2]
        mock_resp = self._mock_response([fake_vec, fake_vec])

        call_count = 0

        def counting_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return 2 embeddings per call (batch_size=2)
            return mock_resp

        with patch("httpx.post", side_effect=counting_post):
            result = provider.embed_texts(["a", "b", "c", "d"])

        # 4 texts / batch_size 2 = 2 calls
        self.assertEqual(call_count, 2)
        self.assertEqual(len(result.vectors), 4)

    def test_expected_dimensions_mismatch_raises(self):
        """If EMBEDDING_DIMENSIONS is set and response differs, raise."""
        from app.services.embeddings.base import EmbeddingDimensionMismatchError

        provider = self._make_provider(expected_dimensions=512)
        # Response returns 3-dim vectors, not 512
        mock_resp = self._mock_response([[0.1, 0.2, 0.3]])

        with patch("httpx.post", return_value=mock_resp):
            with self.assertRaises(EmbeddingDimensionMismatchError) as ctx:
                provider.embed_texts(["test"])
        self.assertEqual(ctx.exception.expected, 512)
        self.assertEqual(ctx.exception.actual, 3)


# ===========================================================================
# Default retrieval path tests
# ===========================================================================

class TestDefaultRetrievalPath(unittest.TestCase):
    """Ensure the default local provider path in retrieval_service is unchanged."""

    def test_compute_text_embedding_shim_returns_correct_dimension(self):
        """compute_text_embedding shim must return 64-dim vector by default."""
        from app.services.knowledge_service import compute_text_embedding

        with patch("app.services.embeddings.factory.settings", _make_settings_override()):
            vec = compute_text_embedding("test query")
        self.assertEqual(len(vec), 64)

    def test_compute_text_embedding_shim_deterministic(self):
        from app.services.knowledge_service import compute_text_embedding

        with patch("app.services.embeddings.factory.settings", _make_settings_override()):
            v1 = compute_text_embedding("same text")
            v2 = compute_text_embedding("same text")
        self.assertEqual(v1, v2)

    def test_compute_text_embedding_shim_empty_text(self):
        from app.services.knowledge_service import compute_text_embedding

        with patch("app.services.embeddings.factory.settings", _make_settings_override()):
            vec = compute_text_embedding("")
        self.assertEqual(len(vec), 64)

    def test_prepare_chunks_uses_provider_factory(self):
        """prepare_chunks_for_indexing must use the provider factory, not raw hash."""
        from app.services.knowledge_service import prepare_chunks_for_indexing
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        chunks = [
            {"chunk_index": 0, "content": "hello world", "token_count": 2, "metadata_json": {}},
            {"chunk_index": 1, "content": "foo bar baz", "token_count": 3, "metadata_json": {}},
        ]

        with patch("app.services.knowledge_service.get_embedding_provider") as mock_factory, \
             patch("app.services.knowledge_service.vector_store") as mock_vs:
            mock_vs.should_store_embeddings_in_database.return_value = False
            mock_provider = LocalHashProvider(model="local-hash-v1", dimensions=64)
            mock_factory.return_value = mock_provider

            result = prepare_chunks_for_indexing(1, "abc123", chunks)

        self.assertEqual(len(result), 2)
        # Verify provider metadata is present in metadata_json
        for item in result:
            meta = item["metadata_json"]
            self.assertIn("embedding_provider", meta)
            self.assertIn("embedding_model", meta)
            self.assertIn("embedding_dimensions", meta)
            self.assertIn("embedding_version", meta)
            self.assertIn("parser_version", meta)
            self.assertIn("chunker_version", meta)

    def test_prepare_chunks_metadata_values(self):
        """Provider metadata values must match local provider."""
        from app.services.knowledge_service import prepare_chunks_for_indexing
        from app.services.embeddings.local_hash_provider import LocalHashProvider

        chunks = [
            {"chunk_index": 0, "content": "test content", "token_count": 2, "metadata_json": {}},
        ]

        with patch("app.services.knowledge_service.get_embedding_provider") as mock_factory, \
             patch("app.services.knowledge_service.vector_store") as mock_vs:
            mock_vs.should_store_embeddings_in_database.return_value = False
            mock_provider = LocalHashProvider(model="local-hash-v1", dimensions=64)
            mock_factory.return_value = mock_provider

            result = prepare_chunks_for_indexing(42, "deadbeef", chunks)

        meta = result[0]["metadata_json"]
        self.assertEqual(meta["embedding_provider"], "local")
        self.assertEqual(meta["embedding_model"], "local-hash-v1")
        self.assertEqual(meta["embedding_dimensions"], 64)
        self.assertEqual(meta["embedding_version"], "local-hash-v1")

    def test_prepare_chunks_empty_input_returns_empty(self):
        from app.services.knowledge_service import prepare_chunks_for_indexing

        result = prepare_chunks_for_indexing(1, "abc", [])
        self.assertEqual(result, [])


# ===========================================================================
# EmbeddingMeta and result type tests
# ===========================================================================

class TestEmbeddingBaseTypes(unittest.TestCase):
    """Sanity checks for base types."""

    def test_embedding_meta_frozen(self):
        from app.services.embeddings.base import EmbeddingMeta

        meta = EmbeddingMeta(provider="local", model="local-hash-v1", dimensions=64, version="local-hash-v1")
        with self.assertRaises((AttributeError, TypeError)):
            meta.provider = "changed"  # type: ignore[misc]

    def test_embed_result_fields(self):
        from app.services.embeddings.base import EmbedResult, EmbeddingMeta

        meta = EmbeddingMeta(provider="local", model="m", dimensions=4, version="v1")
        result = EmbedResult(vectors=[[1.0, 2.0, 3.0, 4.0]], meta=meta)
        self.assertEqual(len(result.vectors), 1)
        self.assertEqual(result.meta.provider, "local")

    def test_query_embed_result_fields(self):
        from app.services.embeddings.base import QueryEmbedResult, EmbeddingMeta

        meta = EmbeddingMeta(provider="local", model="m", dimensions=4, version="v1")
        result = QueryEmbedResult(vector=[1.0, 2.0, 3.0, 4.0], meta=meta)
        self.assertEqual(len(result.vector), 4)

    def test_embedding_provider_error_attributes(self):
        from app.services.embeddings.base import EmbeddingProviderError

        exc = EmbeddingProviderError("test error", provider="ollama", model="m", category="timeout")
        self.assertEqual(exc.provider, "ollama")
        self.assertEqual(exc.model, "m")
        self.assertEqual(exc.category, "timeout")
        self.assertIn("test error", str(exc))

    def test_dimension_mismatch_error_attributes(self):
        from app.services.embeddings.base import EmbeddingDimensionMismatchError

        exc = EmbeddingDimensionMismatchError(
            "dim mismatch", provider="ollama", model="m", expected=512, actual=3
        )
        self.assertEqual(exc.expected, 512)
        self.assertEqual(exc.actual, 3)
        self.assertEqual(exc.category, "dimension_mismatch")


if __name__ == "__main__":
    unittest.main()
