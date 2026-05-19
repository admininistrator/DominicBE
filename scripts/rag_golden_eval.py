#!/usr/bin/env python3
"""RAG golden evaluation — compare local hash, Ollama, and API provider retrieval quality.

This script evaluates retrieval quality against a golden question-answer set.
It works in three modes:

  1. **Current provider (default):** Uses the configured EMBEDDING_PROVIDER to embed
     evaluation queries and scores retrieved-document context for relevance.
     Reports top-k hit rate and citation-usefulness metrics.

  2. **Specific provider (--provider):** Evaluate a specific provider by name
     (e.g. ``--provider local``, ``--provider ollama``, ``--provider api``).

  3. **Provider comparison (--compare):** Evaluate all available providers
     (local, ollama, api) and compare their metrics side by side.

Requirements:
  - The golden set at scripts/data/rag_golden_set.json (10 curated queries).
  - The active embedding provider must be reachable (local is always OK).

Usage:
  python scripts/rag_golden_eval.py
  python scripts/rag_golden_eval.py --provider api
  python scripts/rag_golden_eval.py --compare
  python scripts/rag_golden_eval.py --provider local --compare
  python scripts/rag_golden_eval.py --verbose
  python scripts/rag_golden_eval.py --json  # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from math import sqrt
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rag_golden_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = SCRIPT_DIR / "data" / "rag_golden_set.json"
PROJECT_ROOT = SCRIPT_DIR.parent  # DominicBE/

# ---------------------------------------------------------------------------
# Cosine similarity (same as retrieval_service)
# ---------------------------------------------------------------------------
def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

# P05-T02: expanded _get_provider to support api provider type with config
def _get_provider(provider: str = "local", model: str | None = None):
    """Get an embedding provider by setting environment variables and re-reading.

    Supports ``local``, ``ollama``, and ``api`` provider types.
    """
    import app.core.config as config_module
    from app.services.embeddings.factory import get_embedding_provider

    # Build a mini overrides dict
    if provider == "api":
        model_val = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        dims_val = os.environ.get("EMBEDDING_DIMENSIONS", "1536")
        base_val = os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        coll_val = "knowledge_api_embeddings"
    elif provider == "local":
        model_val = model or "local-hash-v1"
        dims_val = os.environ.get("EMBEDDING_DIMENSIONS", "64")
        base_val = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:11434")
        coll_val = os.environ.get("VECTOR_STORE_COLLECTION", "knowledge_chunks")
    else:  # ollama
        model_val = model or os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        dims_val = os.environ.get("EMBEDDING_DIMENSIONS", "1024")
        base_val = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:11434")
        coll_val = os.environ.get("VECTOR_STORE_COLLECTION", "knowledge_qwen3_embedding_06b")

    overrides = {
        "EMBEDDING_PROVIDER": provider,
        "EMBEDDING_MODEL": model_val,
        "EMBEDDING_DIMENSIONS": dims_val,
        "EMBEDDING_BASE_URL": base_val,
        "EMBEDDING_TIMEOUT_SECONDS": os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "10.0"),
        "VECTOR_STORE_PROVIDER": os.environ.get("VECTOR_STORE_PROVIDER", "qdrant"),
        "VECTOR_STORE_COLLECTION": coll_val,
    }

    # For api provider, also pass through API-specific env vars
    if provider == "api":
        overrides["EMBEDDING_API_KEY"] = os.environ.get("EMBEDDING_API_KEY", "")
        overrides["EMBEDDING_API_TYPE"] = os.environ.get("EMBEDDING_API_TYPE", "openai")
        overrides["EMBEDDING_API_VERSION"] = os.environ.get("EMBEDDING_API_VERSION", "")
        overrides["EMBEDDING_API_HEADERS"] = os.environ.get("EMBEDDING_API_HEADERS", "")

    # Save and restore original env
    saved = {}
    for key, val in overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = val

    try:
        # Force reload settings
        from app.core.config import Settings

        kwargs = {
            "_env_file": None,
            "embedding_provider": provider,
            "embedding_model": overrides["EMBEDDING_MODEL"],
            "embedding_dimensions": int(overrides["EMBEDDING_DIMENSIONS"]),
            "embedding_base_url": overrides["EMBEDDING_BASE_URL"],
            "embedding_timeout_seconds": float(overrides["EMBEDDING_TIMEOUT_SECONDS"]),
            "vector_store_provider": overrides["VECTOR_STORE_PROVIDER"],
            "vector_store_collection": overrides["VECTOR_STORE_COLLECTION"],
        }

        # For api provider, pass through API-specific settings
        if provider == "api":
            kwargs["embedding_api_key"] = overrides["EMBEDDING_API_KEY"]
            kwargs["embedding_api_type"] = overrides["EMBEDDING_API_TYPE"]
            kwargs["embedding_api_version"] = overrides["EMBEDDING_API_VERSION"]
            kwargs["embedding_api_headers"] = overrides["EMBEDDING_API_HEADERS"]

        config_module.settings = Settings(**kwargs)
        return get_embedding_provider()
    except Exception as exc:
        logger.warning("Provider '%s' not available: %s", provider, exc)
        return None
    finally:
        # Restore saved env vars
        for key, val in overrides.items():
            if saved[key] is not None:
                os.environ[key] = saved[key]
            else:
                os.environ.pop(key, None)


def _compute_context_embedding(provider, text: str) -> list[float]:
    """Embed a single context string using the provider."""
    try:
        result = provider.embed_texts([text])
        if result.vectors and len(result.vectors) > 0:
            return result.vectors[0]
    except Exception as exc:
        logger.debug("Context embedding failed: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_provider(provider, provider_name: str, golden_set: list[dict], verbose: bool) -> dict:
    """Run evaluation for a single provider against the golden set.

    Returns a metrics dict with top-k hit rate, average score, latency, etc.
    """
    import app.core.config as config_module

    queries_attempted = 0
    queries_succeeded = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    total_latency_ms = 0
    total_semantic_score = 0.0

    # Embed all golden context texts once for relevance matching
    context_texts = [item["context"] for item in golden_set]
    context_vectors: dict[str, list[float]] = {}
    for idx, item in enumerate(golden_set):
        vec = _compute_context_embedding(provider, item["context"])
        if vec:
            context_vectors[item["id"]] = vec

    for item in golden_set:
        query = item["query"]
        eval_id = item["id"]

        try:
            started = time.perf_counter()
            result = provider.embed_query(query)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            query_vec = result.vector
            queries_attempted += 1
        except Exception as exc:
            if verbose:
                logger.warning("[%s] Query embedding failed: %s", eval_id, exc)
            continue

        if not query_vec or not context_vectors:
            continue

        queries_succeeded += 1
        total_latency_ms += elapsed_ms

        # Score this query against all golden context vectors
        scored: list[tuple[str, float]] = []
        for cid, cvec in context_vectors.items():
            score = _cosine_similarity(query_vec, cvec)
            scored.append((cid, score))

        scored.sort(key=lambda x: -x[1])

        # Top-k hit: is the correct golden set item in the top-k?
        ranked_ids = [sid for sid, _ in scored]
        if eval_id in ranked_ids[:1]:
            top1_hits += 1
        if eval_id in ranked_ids[:3]:
            top3_hits += 1
        if eval_id in ranked_ids[:5]:
            top5_hits += 1

        # Semantic score for the correct item
        for cid, sscore in scored:
            if cid == eval_id:
                total_semantic_score += sscore
                break

    n = queries_succeeded or 1
    # P05-T02: extract api_type if available (from extras or capabilities)
    api_type = ""
    if hasattr(provider, "meta") and hasattr(provider.meta, "extra"):
        api_type = provider.meta.extra.get("api_type", "")
    if not api_type and hasattr(provider, "capabilities"):
        api_type = provider.capabilities.api_type

    metrics = {
        "provider": provider_name,
        "api_type": api_type,
        "model": getattr(provider.meta, "model", "unknown") if hasattr(provider, "meta") else "unknown",
        "dimensions": getattr(provider.meta, "dimensions", 0) if hasattr(provider, "meta") else 0,
        "version": getattr(provider.meta, "version", "") if hasattr(provider, "meta") else "",
        "collection": getattr(config_module.settings, "vector_store_collection", "unknown"),
        "queries_attempted": queries_attempted,
        "queries_succeeded": queries_succeeded,
        "top1_hit_rate": round(top1_hits / n, 4),
        "top3_hit_rate": round(top3_hits / n, 4),
        "top5_hit_rate": round(top5_hits / n, 4),
        "avg_semantic_score": round(total_semantic_score / n, 4),
        "avg_latency_ms": round(total_latency_ms / n, 1) if queries_succeeded else 0,
        "total_latency_ms": total_latency_ms,
    }
    return metrics


def _print_report(results: dict[str, dict], verbose: bool, json_output: bool):
    """Print evaluation report."""
    if json_output:
        print(json.dumps(results, indent=2))
        return

    print("=" * 70)
    print("  RAG Golden Evaluation Report")
    print("=" * 70)

    for provider_name, metrics in results.items():
        skipped = metrics.get("skipped", False)
        skip_reason = metrics.get("skip_reason", "")

        print(f"\n  Provider: {provider_name.upper()}")
        print(f"  Model:    {metrics.get('model', 'N/A')}")
        # P05-T02: show api_type when present
        api_type = metrics.get("api_type", "")
        if api_type:
            print(f"  API type: {api_type}")
        print(f"  Dims:     {metrics.get('dimensions', 'N/A')}")
        print(f"  Version:  {metrics.get('version', 'N/A')}")
        print(f"  Collection: {metrics.get('collection', 'N/A')}")

        if skipped:
            print(f"  Status:   SKIPPED ({skip_reason})")
            continue

        print(f"  Queries attempted:  {metrics['queries_attempted']}")
        print(f"  Queries succeeded:  {metrics['queries_succeeded']}")
        print(f"  Top-1 hit rate:     {metrics['top1_hit_rate']:.2%}")
        print(f"  Top-3 hit rate:     {metrics['top3_hit_rate']:.2%}")
        print(f"  Top-5 hit rate:     {metrics['top5_hit_rate']:.2%}")
        print(f"  Avg semantic score: {metrics['avg_semantic_score']:.4f}")
        print(f"  Avg latency:        {metrics['avg_latency_ms']:.0f} ms/query")
        print(f"  Total latency:      {metrics['total_latency_ms']:.0f} ms")

    if len(results) > 1:
        print()
        print("-" * 70)
        print("  Provider Comparison")
        print("-" * 70)
        providers = [name for name, m in results.items() if not m.get("skipped")]
        if len(providers) >= 2:
            baseline = providers[0]
            candidate = providers[1]
            b = results[baseline]
            c = results[candidate]
            diff_top1 = c["top1_hit_rate"] - b["top1_hit_rate"]
            diff_top3 = c["top3_hit_rate"] - b["top3_hit_rate"]
            diff_latency = c["avg_latency_ms"] - b["avg_latency_ms"]
            print(f"  {candidate} vs {baseline}:")
            print(f"    Top-1 delta:  {diff_top1:+.2%}")
            print(f"    Top-3 delta:  {diff_top3:+.2%}")
            print(f"    Latency delta: {diff_latency:+.0f} ms/query")

            # P05-T02: show dimension delta for cross-provider insight
            b_dims = b.get("dimensions", 0)
            c_dims = c.get("dimensions", 0)
            if b_dims != c_dims:
                print(f"    Dimensions:   {b_dims} (baseline) vs {c_dims} (candidate)")
        else:
            print("  (comparison requires both providers available)")

    print()
    print("=" * 70)
    print("  RAG_GOLDEN_EVAL_DONE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG golden evaluation — compare retrieval quality against a curated question set."
    )
    # P05-T02: --provider flag to evaluate a specific provider
    parser.add_argument("--provider", type=str, default=None,
                        help="Evaluate a specific provider (local, ollama, api). Overrides current config.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available providers (local, ollama, api)")
    parser.add_argument("--verbose", action="store_true", help="Detailed per-query logging")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--golden-set", type=str, default=str(GOLDEN_SET_PATH),
                        help="Path to golden set JSON (default: scripts/data/rag_golden_set.json)")
    args = parser.parse_args()

    # Load golden set
    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        logger.error("Golden set not found at %s", golden_path)
        sys.exit(1)

    with open(golden_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    logger.info("Loaded %d golden evaluation queries from %s", len(golden_set), golden_path)
    if args.verbose:
        for item in golden_set:
            logger.debug("  [%s] %s (difficulty=%s)", item["id"], item["query"], item.get("difficulty", "N/A"))

    results: dict[str, dict] = {}

    # P05-T02: Determine which providers to evaluate
    if args.provider:
        # Single provider specified
        providers_to_eval = [args.provider.strip().lower()]
    else:
        # Detect current provider
        import app.core.config as config_module
        current_provider = (config_module.settings.embedding_provider or "local").strip().lower()
        providers_to_eval = [current_provider]

    # P05-T02: Extended comparison — all supported providers
    if args.compare:
        all_providers = ["local", "ollama", "api"]
        for p in all_providers:
            if p not in providers_to_eval:
                providers_to_eval.append(p)

    for provider_name in providers_to_eval:
        logger.info("Evaluating provider: %s", provider_name)
        provider = _get_provider(provider_name)

        if provider is None:
            results[provider_name] = {
                "provider": provider_name,
                "skipped": True,
                "skip_reason": "provider not available or dependency missing",
            }
            logger.warning("Skipping provider '%s' — not available", provider_name)
            continue

        # Quick connectivity check for non-local providers
        if provider_name != "local":
            try:
                _ = provider.embed_query("connectivity probe")
            except Exception as exc:
                results[provider_name] = {
                    "provider": provider_name,
                    "skipped": True,
                    "skip_reason": f"provider unreachable: {type(exc).__name__}",
                }
                logger.warning("Skipping provider '%s' — unreachable: %s", provider_name, exc)
                continue

        metrics = _evaluate_provider(provider, provider_name, golden_set, args.verbose)
        results[provider_name] = metrics

    if not results:
        logger.error("No providers could be evaluated.")
        sys.exit(1)

    _print_report(results, args.verbose, args.json)


if __name__ == "__main__":
    main()
