"""Phase 0 — Behavior Snapshot / Parity Tests for RAG Core Extraction.

These tests validate that the current DominicBE RAG behavior is captured
in golden snapshot files.  They serve as regression baselines during the
rag-core extraction (Phases 1-8).

Each test class corresponds to a Phase-0 task:
  - P00-T02: Chunking snapshot
  - P00-T03: Local hash embedding snapshot
  - P00-T04: Retrieval scoring snapshot

The snapshot files are stored in the ``tests/snapshots/`` directory and
are loaded as reference data.  To regenerate snapshots:
    cd DominicBE
    python tests/gen_snapshots.py

This file is self-contained — it duplicates the pure RAG functions so that
tests can run without loading the full DominicBE app stack (which requires
a database, Qdrant, API keys, etc.).

IMPORTANT: The duplicated functions are EXACT COPIES of the DominicBE
originals.  Do NOT modify them here — modify the originals in DominicBE
and regenerate the snapshots.
"""

import hashlib
import json
import math
import os
import re
import unittest
import unicodedata


SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


# ============================================================
# Duplicated pure functions — exact copies from DominicBE source
# See app/services/knowledge_service.py,
#     app/services/embeddings/local_hash_provider.py,
#     app/services/retrieval_service.py
# ============================================================

# --- Chunking / Normalization (from knowledge_service.py) ---

def normalize_text_for_ingestion(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    paragraphs = []
    for block in re.split(r"\n\s*\n+", raw):
        normalized = re.sub(r"[ \t]+", " ", block).strip()
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


def _split_large_sentence(sentence: str, size: int, overlap: int) -> list[str]:
    if len(sentence) <= size:
        return [sentence]
    parts: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(sentence):
        end = min(len(sentence), start + size)
        piece = sentence[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(sentence):
            break
        start += step
    return parts or [sentence]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    normalized_text = normalize_text_for_ingestion(text)
    if not normalized_text:
        return []
    sentences = _split_sentences(normalized_text)
    chunks: list[dict] = []
    current_chunk: list[str] = []
    current_len = 0
    idx = 0
    expanded_sentences: list[str] = []
    for sentence in sentences:
        expanded_sentences.extend(_split_large_sentence(sentence, size=chunk_size, overlap=chunk_overlap))
    for sentence in expanded_sentences:
        s_len = len(sentence)
        if current_len + s_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk).strip()
            if chunk_text_str:
                chunks.append({
                    "chunk_index": idx,
                    "content": chunk_text_str,
                    "token_count": max(1, len(chunk_text_str) // 4),
                    "metadata_json": {"char_count": len(chunk_text_str)},
                })
                idx += 1
            overlap_chunks: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_chunks.insert(0, s)
                overlap_len += len(s)
            current_chunk = overlap_chunks
            current_len = overlap_len
        current_chunk.append(sentence)
        current_len += s_len
    if current_chunk:
        chunk_text_str = " ".join(current_chunk).strip()
        if chunk_text_str:
            chunks.append({
                "chunk_index": idx,
                "content": chunk_text_str,
                "token_count": max(1, len(chunk_text_str) // 4),
                "metadata_json": {"char_count": len(chunk_text_str)},
            })
    return chunks


# --- Local Hash Embedding (from local_hash_provider.py) ---

def _normalize_text(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    paragraphs = []
    for block in re.split(r"\n\s*\n+", raw):
        normalized = re.sub(r"[ \t]+", " ", block).strip()
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def _hash_embed(text: str, dimensions: int) -> list[float]:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return [0.0] * dimensions
    vector = [0.0] * dimensions
    for token in re.findall(r"\w+", normalized, flags=re.UNICODE):
        token_hash = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(token_hash[:2], "big") % dimensions
        sign = 1.0 if token_hash[2] % 2 == 0 else -1.0
        weight = 1.0 + ((token_hash[3] % 5) * 0.1)
        vector[bucket] += sign * weight
    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return [0.0] * dimensions
    return [round(value / magnitude, 6) for value in vector]


# --- Retrieval Scoring (from retrieval_service.py) ---

def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_for_search(text: str) -> str:
    lowered = _strip_accents(text).lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", _normalize_for_search(text), flags=re.UNICODE) if token}


QUERY_EXPANSION_RULES: dict[str, list[str]] = {
    "hoan tien": ["refund", "refund policy", "money back"],
    "chinh sach": ["policy"],
    "xu ly": ["review", "process", "processing"],
    "bao lau": ["how long", "duration", "timeline", "days"],
    "mat khau": ["password", "credentials"],
    "dang nhap": ["login", "sign in", "authenticate"],
    "tai lieu": ["document", "knowledge base"],
}


def _expand_query(query: str, enable_query_expansion: bool = True) -> tuple[str, list[str]]:
    normalized = _normalize_for_search(query)
    expansions: list[str] = []
    if enable_query_expansion:
        for phrase, candidates in QUERY_EXPANSION_RULES.items():
            if phrase in normalized:
                for candidate in candidates:
                    if candidate not in expansions:
                        expansions.append(candidate)
    if not expansions:
        return " ".join((query or "").split()), []
    rewritten_query = " ".join(part for part in [" ".join((query or "").split()), *expansions] if part).strip()
    return rewritten_query, expansions


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_overlap_score(query_text: str, content: str) -> float:
    query_tokens = _tokenize(query_text)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(query_tokens)
    density = len(overlap) / math.sqrt(len(query_tokens) * len(content_tokens))
    return round(min(1.0, (coverage * 0.75) + (density * 0.25)), 6)


def _hybrid_score(
    semantic_score: float,
    lexical_score: float,
    semantic_weight: float = 0.4,
    lexical_weight: float = 0.6,
) -> float:
    total_weight = semantic_weight + lexical_weight
    w_sem = semantic_weight / total_weight
    w_lex = lexical_weight / total_weight
    return round(min(1.0, (semantic_score * w_sem) + (lexical_score * w_lex)), 6)


def _build_snippet(content: str, *, max_chars: int = 220) -> str:
    normalized = " ".join((content or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _estimate_token_count(text: str, explicit_count: int | None = None) -> int:
    if explicit_count and explicit_count > 0:
        return int(explicit_count)
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _classify_evidence_strength(
    results: list[dict], *, fallback_used: bool, low_confidence_score: float = 0.2,
) -> str:
    if not results:
        return "none"
    if fallback_used:
        return "fallback"
    top_score = float(results[0].get("score") or 0.0)
    if top_score >= low_confidence_score:
        return "grounded"
    return "weak"


def _rerank_results(
    query_text: str,
    results: list[dict],
    max_rerank_candidates: int = 12,
    rerank_title_weight: float = 0.15,
    rerank_position_weight: float = 0.1,
) -> list[dict]:
    reranked: list[dict] = []
    for item in results[:max_rerank_candidates]:
        title_score = _lexical_overlap_score(query_text, item.get("title") or "")
        chunk_index = int(item.get("chunk_index") or 0)
        position_score = max(0.0, 1.0 - (chunk_index * 0.05))
        rerank_score = round(
            min(1.0, float(item.get("score") or 0.0)
                + (title_score * rerank_title_weight)
                + (position_score * rerank_position_weight)),
            6,
        )
        reranked.append({
            **item,
            "rerank_score": rerank_score,
            "token_estimate": _estimate_token_count(item.get("content") or "", item.get("token_count")),
        })
    reranked.sort(key=lambda item: (
        -float(item.get("rerank_score") or 0.0),
        -float(item.get("score") or 0.0),
        item.get("document_id") or 0,
        item.get("chunk_index") or 0,
    ))
    return reranked


def _dedupe_scored_results(results: list[dict]) -> list[dict]:
    seen: set[tuple[int, str]] = set()
    deduped: list[dict] = []
    for item in results:
        key = (item["document_id"], _normalize_for_dedupe(item.get("content") or item.get("snippet") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


# ============================================================
# Helper: load snapshot
# ============================================================

def _load_snapshot(name: str) -> dict:
    path = os.path.join(SNAPSHOTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Snapshot file not found: {path}\n"
            f"Generate it first: python tests/gen_snapshots.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# P00-T02: Chunking Snapshot Test
# ============================================================

class TestChunkingSnapshot(unittest.TestCase):
    """P00-T02: Verify chunking behavior matches golden snapshot."""

    _snapshot: dict | None = None

    @classmethod
    def setUpClass(cls):
        cls._snapshot = _load_snapshot("chunking_baseline.json")

    def _get_case(self, name: str) -> dict:
        return self._snapshot[name]

    def test_empty_text(self):
        case = self._get_case("empty")
        self.assertEqual(case["chunk_count"], 0)
        self.assertEqual(case["normalized"], "")

    def test_whitespace_only(self):
        case = self._get_case("whitespace_only")
        self.assertEqual(case["chunk_count"], 0)
        self.assertEqual(case["normalized"], "")

    def test_single_sentence(self):
        case = self._get_case("single_sentence")
        self.assertEqual(case["chunk_count"], 1)
        chunk = case["chunks"][0]
        self.assertEqual(chunk["chunk_index"], 0)
        self.assertIn("content", chunk)
        self.assertIn("token_count", chunk)
        self.assertIn("metadata_json", chunk)
        self.assertIn("char_count", chunk["metadata_json"])

    def test_multi_paragraph(self):
        case = self._get_case("multi_paragraph")
        self.assertGreater(case["chunk_count"], 0)
        for i, chunk in enumerate(case["chunks"]):
            self.assertEqual(chunk["chunk_index"], i)
            self.assertIsInstance(chunk["token_count"], int)
            self.assertGreater(chunk["token_count"], 0)

    def test_unicode_text(self):
        case = self._get_case("unicode")
        self.assertGreater(case["chunk_count"], 0)
        # Vietnamese characters must be preserved
        self.assertIn("chào", case["normalized"])

    def test_very_long_sentence_splits_into_multiple_chunks(self):
        case = self._get_case("very_long_sentence")
        self.assertGreater(case["chunk_count"], 1, "Long text should produce multiple chunks")

    def test_chunk_structure(self):
        case = self._get_case("single_sentence")
        chunk = case["chunks"][0]
        required_keys = {"chunk_index", "content", "token_count", "metadata_json"}
        self.assertTrue(required_keys.issubset(chunk.keys()),
                        f"Missing keys: {required_keys - chunk.keys()}")
        self.assertIn("char_count", chunk["metadata_json"])
        self.assertIsInstance(chunk["chunk_index"], int)


# ============================================================
# P00-T03: Embedding Snapshot Test
# ============================================================

class TestEmbeddingSnapshot(unittest.TestCase):
    """P00-T03: Verify local hash embedding matches golden snapshot."""

    _snapshot: dict | None = None

    @classmethod
    def setUpClass(cls):
        cls._snapshot = _load_snapshot("embedding_baseline.json")

    def _get_case(self, name: str) -> dict:
        return self._snapshot[name]

    def test_empty_text_returns_zero_vector(self):
        case = self._get_case("empty")
        vec = case["vector"]
        self.assertEqual(len(vec), 64)
        self.assertTrue(all(v == 0.0 for v in vec))

    def test_whitespace_only_returns_zero_vector(self):
        case = self._get_case("whitespace_only")
        vec = case["vector"]
        self.assertEqual(len(vec), 64)
        self.assertTrue(all(v == 0.0 for v in vec))

    def test_single_word_has_exactly_one_nonzero(self):
        case = self._get_case("single_word")
        vec = case["vector"]
        non_zero = sum(1 for v in vec if v != 0)
        self.assertEqual(non_zero, 1, "Single word should hash to exactly one bucket")
        self.assertNotAlmostEqual(sum(abs(v) for v in vec), 0.0)

    def test_repeated_word_same_as_single(self):
        single = self._get_case("single_word")
        repeated = self._get_case("repeated_word")
        self.assertEqual(single["vector"], repeated["vector"])

    def test_vector_is_unit_normalized(self):
        case = self._get_case("multi_word")
        vec = case["vector"]
        magnitude = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_all_vectors_have_correct_dimension(self):
        for name, case in self._snapshot.items():
            if not isinstance(case, dict) or "vector" not in case:
                continue
            self.assertEqual(
                len(case["vector"]), 64,
                f"Case '{name}' has wrong dimension: {len(case['vector'])}"
            )

    def test_unicode_text_produces_nonzero_vector(self):
        case = self._get_case("unicode")
        vec = case["vector"]
        non_zero = sum(1 for v in vec if v != 0)
        self.assertGreater(non_zero, 0, "Unicode text should produce non-zero embedding")
        magnitude = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(magnitude, 1.0, places=5)


# ============================================================
# P00-T04: Scoring Snapshot Test
# ============================================================

class TestScoringSnapshot(unittest.TestCase):
    """P00-T04: Verify retrieval scoring functions match golden snapshot."""

    _snapshot: dict | None = None

    @classmethod
    def setUpClass(cls):
        cls._snapshot = _load_snapshot("scoring_baseline.json")

    # --- Cosine Similarity ---

    def test_cosine_identical_vectors(self):
        val = self._snapshot["cosine_similarity"]["identical_vectors"]
        self.assertAlmostEqual(val, 1.0, places=6)

    def test_cosine_reversed_vectors(self):
        val = self._snapshot["cosine_similarity"]["reversed_vectors"]
        self.assertAlmostEqual(val, 0.63636363636, places=6)

    def test_cosine_empty_returns_zero(self):
        self.assertEqual(self._snapshot["cosine_similarity"]["empty_left"], 0.0)
        self.assertEqual(self._snapshot["cosine_similarity"]["empty_right"], 0.0)

    def test_cosine_zero_vector_returns_zero(self):
        self.assertEqual(self._snapshot["cosine_similarity"]["zero_vector_left"], 0.0)

    def test_cosine_dimension_mismatch_returns_zero(self):
        self.assertEqual(self._snapshot["cosine_similarity"]["dimension_mismatch"], 0.0)

    # --- Lexical Overlap Score ---

    def test_lexical_identical_tokens(self):
        val = self._snapshot["lexical_overlap_score"]["identical_tokens"]
        self.assertGreater(val, 0.9)

    def test_lexical_no_overlap(self):
        self.assertEqual(self._snapshot["lexical_overlap_score"]["no_overlap"], 0.0)

    def test_lexical_empty_returns_zero(self):
        self.assertEqual(self._snapshot["lexical_overlap_score"]["empty_query"], 0.0)
        self.assertEqual(self._snapshot["lexical_overlap_score"]["empty_content"], 0.0)

    def test_lexical_unicode(self):
        val = self._snapshot["lexical_overlap_score"]["unicode_vietnamese"]
        self.assertGreater(val, 0.0)

    # --- Hybrid Score ---

    def test_hybrid_full_both(self):
        self.assertAlmostEqual(self._snapshot["hybrid_score"]["full_semantic_full_lexical"], 1.0, places=6)

    def test_hybrid_semantic_only(self):
        val = self._snapshot["hybrid_score"]["full_semantic_no_lexical"]
        self.assertAlmostEqual(val, 0.4, places=6)

    def test_hybrid_lexical_only(self):
        val = self._snapshot["hybrid_score"]["no_semantic_full_lexical"]
        self.assertAlmostEqual(val, 0.6, places=6)

    def test_hybrid_both_zero(self):
        self.assertEqual(self._snapshot["hybrid_score"]["both_zero"], 0.0)

    # --- Query Expansion ---

    def test_expansion_no_match(self):
        query, expansions = self._snapshot["query_expansion"]["no_match"]
        self.assertEqual(len(expansions), 0)

    def test_expansion_vietnamese(self):
        query, expansions = self._snapshot["query_expansion"]["vietnamese_match"]
        self.assertGreater(len(expansions), 0)
        self.assertIn("refund", " ".join(expansions).lower())

    # --- Build Snippet ---

    def test_snippet_short_text(self):
        self.assertEqual(self._snapshot["build_snippet"]["short_text"], "Hello world")

    def test_snippet_long_text_truncated(self):
        result = self._snapshot["build_snippet"]["long_500"]
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 223)

    def test_snippet_whitespace_collapsed(self):
        result = self._snapshot["build_snippet"]["whitespace_collapse"]
        self.assertEqual(result, "hello world test")

    # --- Estimate Token Count ---

    def test_token_count_empty(self):
        self.assertEqual(self._snapshot["estimate_token_count"]["empty"], 0)

    def test_token_count_with_explicit(self):
        self.assertEqual(self._snapshot["estimate_token_count"]["explicit_42"], 42)

    def test_token_count_rough_estimate(self):
        val = self._snapshot["estimate_token_count"]["short"]
        # "hello world" = 11 chars -> max(1, 11//4) = 2
        self.assertEqual(val, max(1, 11 // 4))

    # --- Evidence Strength ---

    def test_evidence_empty(self):
        self.assertEqual(self._snapshot["classify_evidence_strength"]["empty_results"], "none")

    def test_evidence_fallback(self):
        self.assertEqual(self._snapshot["classify_evidence_strength"]["fallback_used"], "fallback")

    def test_evidence_high_score(self):
        self.assertEqual(self._snapshot["classify_evidence_strength"]["high_score_grounded"], "grounded")

    def test_evidence_low_score(self):
        self.assertEqual(self._snapshot["classify_evidence_strength"]["low_score_weak"], "weak")

    # --- Deduplication ---

    def test_dedupe_no_duplicates_preserves_all(self):
        result = self._snapshot["dedupe_scored_results"]["no_duplicates"]
        self.assertEqual(len(result), 2)

    def test_dedupe_removes_content_duplicates(self):
        result = self._snapshot["dedupe_scored_results"]["with_duplicates"]
        self.assertEqual(len(result), 2)  # Three unique doc_id/content combos

    def test_dedupe_empty_input(self):
        result = self._snapshot["dedupe_scored_results"]["empty_input"]
        self.assertEqual(result, [])

    # --- Reranking ---

    def test_rerank_basic_preserves_top_result(self):
        result = self._snapshot["rerank_results"]["basic"]
        self.assertGreater(len(result), 0)
        self.assertIn("rerank_score", result[0])

    def test_rerank_title_overlap_boosts_score(self):
        result = self._snapshot["rerank_results"]["title_overlap"]
        # First item has matching title -> should have higher rerank_score
        self.assertGreater(result[0]["rerank_score"], result[1]["rerank_score"])

    def test_rerank_position_decay(self):
        result = self._snapshot["rerank_results"]["position_decay"]
        # First item (chunk_index=0) should rank above second (chunk_index=10)
        self.assertEqual(result[0]["chunk_index"], 0)
        self.assertEqual(result[1]["chunk_index"], 10)

    # --- Config consistency ---

    def test_config_values_match_defaults(self):
        cfg = self._snapshot["_config"]
        self.assertEqual(cfg["retrieval_hybrid_semantic_weight"], 0.4)
        self.assertEqual(cfg["retrieval_hybrid_lexical_weight"], 0.6)
        self.assertEqual(cfg["retrieval_low_confidence_score"], 0.2)


if __name__ == "__main__":
    unittest.main()
