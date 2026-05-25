"""Standalone snapshot generator for Phase 0 baseline discovery.

This script duplicates the pure RAG functions inline to avoid loading
the full DominicBE application stack (DB models, SQLAlchemy, settings).

The duplicated functions are EXACT copies of the originals in:
  - app/services/knowledge_service.py  (chunking, normalization)
  - app/services/embeddings/local_hash_provider.py  (embedding)
  - app/services/retrieval_service.py  (scoring, reranking, etc.)

Usage:
    cd DominicBE
    .venv\Scripts\python.exe tests/gen_snapshots.py
"""

import hashlib
import json
import math
import os
import re
import unicodedata


# ============================================================
# Duplicated pure functions — exact copies from DominicBE source
# ============================================================

# --- From app/services/knowledge_service.py ---

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


# --- From app/services/embeddings/local_hash_provider.py ---

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


# --- From app/services/retrieval_service.py ---

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


def _classify_evidence_strength(results: list[dict], *, fallback_used: bool, low_confidence_score: float = 0.2) -> str:
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
# Snapshot Generators
# ============================================================

SNAPSHOTS_DIR = "tests/snapshots"


def gen_chunking_snapshot():
    inputs = {
        "empty": "",
        "whitespace_only": "   \n\n  \t  ",
        "single_sentence": "Hello world. This is a test.",
        "multi_paragraph": (
            "First paragraph about chunking.\n\n"
            "Second paragraph about embedding.\n\n"
            "Third paragraph about retrieval."
        ),
        "unicode": (
            "Xin chào thế giới. Đây là văn bản tiếng Việt. "
            "Chúng tôi đang kiểm tra chunking."
        ),
        "very_long_sentence": "Word " * 500,
    }
    snapshot = {}
    for name, text in inputs.items():
        normalized = normalize_text_for_ingestion(text)
        chunks = chunk_text(text, chunk_size=800, chunk_overlap=100)
        snapshot[name] = {
            "input_length": len(text),
            "normalized": normalized,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    path = f"{SNAPSHOTS_DIR}/chunking_baseline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"Chunking snapshot -> {path}  ({len(json.dumps(snapshot))} bytes)")
    for name, data in snapshot.items():
        print(f"  {name}: {data['chunk_count']} chunks")
    return snapshot


def gen_embedding_snapshot():
    inputs = {
        "empty": "",
        "whitespace_only": "   \n\n  \t  ",
        "single_word": "hello",
        "multi_word": "Hello world this is a test sentence for embedding",
        "unicode": "Xin chào thế giới. Đây là văn bản tiếng Việt.",
        "repeated_word": "hello hello hello hello hello",
    }
    snapshot = {}
    for name, text in inputs.items():
        vec = _hash_embed(text, 64)
        snapshot[name] = {
            "input": text,
            "input_length": len(text),
            "vector": vec,
            "vector_length": len(vec),
        }
    path = f"{SNAPSHOTS_DIR}/embedding_baseline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"\nEmbedding snapshot -> {path}  ({len(json.dumps(snapshot))} bytes)")
    for name, data in snapshot.items():
        vec = data["vector"]
        non_zero = sum(1 for v in vec if v != 0)
        print(f"  {name}: sum={sum(vec):.6f} non-zero={non_zero}/{len(vec)}")
    return snapshot


def gen_scoring_snapshot():
    vec_a = [0.1, 0.2, 0.3, 0.4, 0.5]
    vec_b = [0.1, 0.2, 0.3, 0.4, 0.5]
    vec_c = [0.5, 0.4, 0.3, 0.2, 0.1]
    vec_d = [1.0, 0.0, 0.0, 0.0, 0.0]
    vec_zero = [0.0, 0.0, 0.0, 0.0, 0.0]

    snapshot = {
        "_config": {
            "retrieval_hybrid_semantic_weight": 0.4,
            "retrieval_hybrid_lexical_weight": 0.6,
            "retrieval_low_confidence_score": 0.2,
            "retrieval_max_rerank_candidates": 12,
            "retrieval_rerank_title_weight": 0.15,
            "retrieval_rerank_position_weight": 0.1,
        },
        "cosine_similarity": {
            "identical_vectors": _cosine_similarity(vec_a, vec_b),
            "reversed_vectors": _cosine_similarity(vec_a, vec_c),
            "orthogonal_ish": _cosine_similarity(vec_a, vec_d),
            "empty_left": _cosine_similarity([], vec_b),
            "empty_right": _cosine_similarity(vec_a, []),
            "zero_vector_left": _cosine_similarity(vec_zero, vec_b),
            "dimension_mismatch": _cosine_similarity(vec_a, [0.1, 0.2]),
        },
        "lexical_overlap_score": {
            "identical_tokens": _lexical_overlap_score("hello world", "hello world"),
            "partial_overlap": _lexical_overlap_score("hello world foo", "hello world bar"),
            "no_overlap": _lexical_overlap_score("abc def", "ghi jkl"),
            "empty_query": _lexical_overlap_score("", "hello world"),
            "empty_content": _lexical_overlap_score("hello world", ""),
            "unicode_vietnamese": _lexical_overlap_score("xin chào thế giới", "xin chào bạn"),
        },
        "hybrid_score": {
            "full_semantic_full_lexical": _hybrid_score(1.0, 1.0),
            "full_semantic_no_lexical": _hybrid_score(1.0, 0.0),
            "no_semantic_full_lexical": _hybrid_score(0.0, 1.0),
            "half_and_half": _hybrid_score(0.5, 0.5),
            "both_zero": _hybrid_score(0.0, 0.0),
        },
        "query_expansion": {
            "no_match": _expand_query("hello world"),
            "vietnamese_match": _expand_query("hoan tien policy"),
            "multi_rule": _expand_query("chinh sach va hoan tien"),
        },
        "normalize_for_search": {
            "basic": _normalize_for_search("Hello World!"),
            "accents": _normalize_for_search("Xin chào thế giới"),
            "punctuation": _normalize_for_search("Hello... World?! Test."),
        },
        "tokenize": {
            "basic": sorted(_tokenize("Hello World!")),
            "vietnamese": sorted(_tokenize("Xin chào thế giới")),
            "empty": sorted(_tokenize("")),
        },
        "build_snippet": {
            "short_text": _build_snippet("Hello world"),
            "exact_220": _build_snippet("x" * 220),
            "long_500": _build_snippet("x" * 500),
            "whitespace_collapse": _build_snippet("   hello    world   test   "),
        },
        "estimate_token_count": {
            "empty": _estimate_token_count(""),
            "short": _estimate_token_count("hello world"),
            "explicit_42": _estimate_token_count("hello world", explicit_count=42),
            "explicit_zero": _estimate_token_count("hello world", explicit_count=0),
        },
        "classify_evidence_strength": {
            "empty_results": _classify_evidence_strength([], fallback_used=False),
            "fallback_used": _classify_evidence_strength([{"score": 0.5}], fallback_used=True),
            "high_score_grounded": _classify_evidence_strength([{"score": 0.3}], fallback_used=False),
            "low_score_weak": _classify_evidence_strength([{"score": 0.1}], fallback_used=False),
            "score_zero": _classify_evidence_strength([{"score": 0.0}], fallback_used=False),
        },
        "dedupe_scored_results": {
            "no_duplicates": _dedupe_scored_results([
                {"document_id": 1, "content": "hello world"},
                {"document_id": 2, "content": "foo bar"},
            ]),
            "with_duplicates": _dedupe_scored_results([
                {"document_id": 1, "content": "hello world"},
                {"document_id": 1, "content": "Hello   World"},
                {"document_id": 2, "content": "foo bar"},
                {"document_id": 1, "content": "hello world"},
            ]),
            "empty_input": _dedupe_scored_results([]),
        },
        "rerank_results": {
            "basic": _rerank_results("hello", [
                {"score": 0.5, "title": "hello world", "chunk_index": 0,
                 "document_id": 1, "content": "some content"},
                {"score": 0.3, "title": "foo bar", "chunk_index": 2,
                 "document_id": 1, "content": "other content"},
            ]),
            "title_overlap": _rerank_results("exact title match", [
                {"score": 0.1, "title": "exact title match", "chunk_index": 0,
                 "document_id": 1, "content": "content a"},
                {"score": 0.1, "title": "unrelated title", "chunk_index": 1,
                 "document_id": 1, "content": "content b"},
            ]),
            "position_decay": _rerank_results("test", [
                {"score": 0.2, "title": "", "chunk_index": 0,
                 "document_id": 1, "content": "content a"},
                {"score": 0.2, "title": "", "chunk_index": 10,
                 "document_id": 1, "content": "content b"},
            ]),
        },
    }

    path = f"{SNAPSHOTS_DIR}/scoring_baseline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"\nScoring snapshot -> {path}  ({len(json.dumps(snapshot))} bytes)")
    print(f"  cosine identical={snapshot['cosine_similarity']['identical_vectors']}")
    print(f"  hybrid full/full={snapshot['hybrid_score']['full_semantic_full_lexical']}")
    return snapshot


if __name__ == "__main__":
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    gen_chunking_snapshot()
    gen_embedding_snapshot()
    gen_scoring_snapshot()
    print("\nAll snapshots generated successfully")
