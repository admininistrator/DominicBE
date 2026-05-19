#!/usr/bin/env python3
"""RAG operational diagnostics — inspect embedding, vector store, and ingestion state.

This script provides actionable diagnostics for a local-first production-like RAG setup.
It is non-mutating by default (read-only).  No raw document text, vectors, API keys,
or secrets are printed.

Diagnostic categories (output by default):
  1. Active embedding provider, model, dimensions, base URL.
  2. Embedding health check result (in-process call, no server needed).
  3. Vector store provider, active collection, Qdrant reachability, dimension info.
  4. Provider metadata counts across indexed chunks (by provider, model, dimensions).
  5. Documents/chunks needing reindex (missing or mismatched provider metadata).
  6. Recent retrieval event health (returned count, fallback flag, mixed-space skips).

Usage:
  python scripts/diagnose_rag.py
  python scripts/diagnose_rag.py --json    # machine-readable JSON output
  python scripts/diagnose_rag.py --verbose  # detailed per-document chunk breakdown

Exit codes:
  0 — All diagnostics collected (may include warnings but no hard failures).
  1 — Connection or dependency failure prevented diagnostics from completing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("diagnose_rag")

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # scripts/ is sibling of app/

if not (PROJECT_ROOT / "app").is_dir():
    # If run from DominicBE/, adjust
    PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _check_embedding_health_local() -> dict:
    """In-process embedding health check (same logic as app/main.py)."""
    try:
        from app.core.config import settings
    except ImportError as exc:
        return {"ok": False, "detail": f"cannot import settings: {exc}"}

    provider = (settings.embedding_provider or "local").strip().lower()
    model = settings.embedding_model or "local-hash-v1"
    base_url = (settings.embedding_base_url or "http://localhost:11434").rstrip("/")
    # P05-T03: capture api_type for diagnostics
    api_type = (settings.embedding_api_type or "").strip().lower()
    api_type_str = api_type if api_type else ""

    if provider == "local":
        return {
            "ok": True,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_type": api_type_str,
            "detail": "local provider requires no external service",
        }

    if provider == "ollama":
        tags_url = f"{base_url}/api/tags"
        try:
            import httpx
            timeout = min(float(settings.embedding_timeout_seconds or 60.0), 10.0)
            resp = httpx.get(tags_url, timeout=timeout)
            if resp.is_success:
                data = resp.json()
                models_raw = data.get("models") or []
                available = [
                    (entry.get("name") or entry.get("model") or "")
                    for entry in models_raw
                ]
                model_found = any(model in name or name in model for name in available)
                return {
                    "ok": True,
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "api_type": api_type_str,
                    "model_listed": model_found,
                    "detail": "reachable" + ("" if model_found else f"; model '{model}' not pulled"),
                }
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url,
                    "api_type": api_type_str,
                    "detail": f"HTTP {resp.status_code}"}
        except ImportError:
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url,
                    "api_type": api_type_str,
                    "detail": "httpx not installed"}
        except Exception as exc:
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url,
                    "api_type": api_type_str,
                    "detail": f"unavailable: {type(exc).__name__}"}

    # P05-T03: API provider health check — probe with a minimal embedding request
    if provider == "api":
        try:
            from app.services.embeddings.generic_api_provider import GenericAPIProvider

            probe_provider = GenericAPIProvider(
                model=model,
                base_url=base_url,
                api_key=settings.embedding_api_key or "",
                api_type=api_type,
                timeout_seconds=min(float(settings.embedding_timeout_seconds or 60.0), 10.0),
                batch_size=1,
                expected_dimensions=int(settings.embedding_dimensions or 0),
                api_version=settings.embedding_api_version or "",
            )
            probe_start = time.time()
            probe_result = probe_provider.embed_query("health probe")
            probe_latency = int((time.perf_counter() - probe_start) * 1000)
            detected_dims = len(probe_result.vector) if probe_result.vector else 0
            return {
                "ok": True,
                "provider": provider,
                "model": model,
                "base_url": base_url.split("@")[-1].rstrip("/"),
                "api_type": api_type,
                "dimensions": detected_dims,
                "latency_ms": probe_latency,
                "detail": "reachable",
            }
        except ImportError as exc:
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url,
                    "api_type": api_type,
                    "detail": f"generic_api_provider import error: {exc}"}
        except Exception as exc:
            return {"ok": False, "provider": provider, "model": model, "base_url": base_url,
                    "api_type": api_type,
                    "detail": f"unavailable: {type(exc).__name__}"}

    return {"ok": False, "provider": provider, "api_type": api_type_str, "detail": f"unknown provider: {provider}"}


def _try_qdrant_health() -> dict:
    """Check Qdrant reachability and collection state."""
    try:
        from app.core.config import settings
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import UnexpectedResponse

        url = getattr(settings, "vector_store_url", None) or os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url, timeout=5.0)
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        active_collection = settings.vector_store_collection
        active_info = None
        if active_collection in collection_names:
            info = client.get_collection(active_collection)
            dims = info.config.params.vectors.size
            point_count = info.points_count
            active_info = {"collection": active_collection, "dimensions": dims, "points": point_count}

        return {
            "ok": True,
            "url": url.rsplit("@", 1)[-1] if "@" in url else url,  # mask credentials
            "collections": collection_names,
            "active_collection": active_collection,
            "active_collection_info": active_info,
            "detail": "reachable",
        }
    except ImportError:
        return {"ok": False, "detail": "qdrant-client not installed"}
    except Exception as exc:
        return {"ok": False, "detail": f"unreachable: {type(exc).__name__}"}


def _try_db_health() -> dict:
    """Check database reachability."""
    try:
        from app.core.database import SessionLocal, check_database_health
        db = SessionLocal()
        healthy = check_database_health(db)
        db.close()
        return {"ok": healthy}
    except Exception as exc:
        return {"ok": False, "detail": f"unreachable: {type(exc).__name__}"}


def _query_provider_metadata_counts(db_session) -> list[dict]:
    """Query provider metadata distribution across chunks."""
    from app.crud import crud_knowledge
    from app.core.json_utils import ensure_json_mapping
    from app.models.knowledge_models import KnowledgeChunk as Chunk
    from sqlalchemy import func

    if not db_session:
        return []

    try:
        rows = db_session.query(
            Chunk.metadata_json,
            func.count(Chunk.id).label("cnt"),
        ).group_by(Chunk.metadata_json).all()

        counts: dict[str, dict] = {}
        for row in rows:
            meta = ensure_json_mapping(row[0]) if row[0] else {}
            provider = meta.get("embedding_provider", "unknown")
            model = meta.get("embedding_model", "unknown")
            dims = str(meta.get("embedding_dimensions", "unknown"))
            version = meta.get("embedding_version", "")
            parser = meta.get("parser_version", "")
            chunker = meta.get("chunker_version", "")
            key = f"{provider}/{model}/{dims}"
            if key not in counts:
                counts[key] = {
                    "provider": provider,
                    "model": model,
                    "dimensions": dims,
                    "embedding_version": version,
                    "parser_version": parser,
                    "chunker_version": chunker,
                    "chunk_count": 0,
                }
            counts[key]["chunk_count"] += row[1]

        return sorted(counts.values(), key=lambda x: -x["chunk_count"])
    except Exception as exc:
        logger.warning("Cannot query provider metadata: %s", exc)
        return []


def _find_documents_needing_reindex(db_session, provider_meta: list[dict]) -> list[dict]:
    """Find documents/chunks missing or mismatched provider metadata."""
    from app.models.knowledge_models import Document, KnowledgeChunk as Chunk
    from app.core.config import settings
    from app.core.json_utils import ensure_json_mapping
    from sqlalchemy import func

    if not db_session:
        return []

    try:
        current_provider = (settings.embedding_provider or "local").strip().lower()
        current_model = settings.embedding_model or "local-hash-v1"

        # Find chunks with missing or mismatched provider metadata
        mismatched_docs: dict[int, dict] = {}

        # Use a limited samples approach: query recent documents with their chunks
        docs = db_session.query(Document).order_by(Document.id.desc()).limit(100).all()
        doc_ids = [d.id for d in docs]
        if not doc_ids:
            return []

        chunks = db_session.query(Chunk).filter(Chunk.document_id.in_(doc_ids)).all()
        for chunk in chunks:
            meta = ensure_json_mapping(chunk.metadata_json) if chunk.metadata_json else {}
            chunk_provider = (meta.get("embedding_provider") or "").strip().lower()
            chunk_model = (meta.get("embedding_model") or "").strip().lower()
            needs_reindex = False
            reasons: list[str] = []

            if not chunk_provider:
                needs_reindex = True
                reasons.append("missing_provider")
            elif chunk_provider != current_provider:
                needs_reindex = True
                reasons.append(f"provider_mismatch({chunk_provider} != {current_provider})")

            if not chunk_model:
                needs_reindex = True
                reasons.append("missing_model")
            elif chunk_model != current_model:
                needs_reindex = True
                reasons.append(f"model_mismatch({chunk_model} != {current_model})")

            if needs_reindex:
                doc_id = chunk.document_id
                if doc_id not in mismatched_docs:
                    doc = next((d for d in docs if d.id == doc_id), None)
                    mismatched_docs[doc_id] = {
                        "document_id": doc_id,
                        "title": doc.title if doc else f"doc_{doc_id}",
                        "owner": doc.owner_username if doc else "unknown",
                        "reasons": set(),
                    }
                for r in reasons:
                    mismatched_docs[doc_id]["reasons"].add(r)

        result = []
        for doc_id, info in mismatched_docs.items():
            result.append({
                "document_id": doc_id,
                "title": info["title"],
                "owner": info["owner"],
                "reasons": sorted(info["reasons"]),
            })

        return sorted(result, key=lambda x: x["document_id"])
    except Exception as exc:
        logger.warning("Cannot find documents needing reindex: %s", exc)
        return []


def _recent_retrieval_events(db_session, limit: int = 20) -> list[dict]:
    """Query recent retrieval events."""
    from app.models.knowledge_models import RetrievalEvent
    from app.core.json_utils import ensure_json_mapping
    if not db_session:
        return []
    try:
        events = db_session.query(RetrievalEvent).order_by(RetrievalEvent.id.desc()).limit(limit).all()
        result = []
        for ev in events:
            meta = ensure_json_mapping(ev.metadata_json) if ev.metadata_json else {}
            result.append({
                "id": ev.id,
                "request_id": ev.request_id,
                "username": ev.username,
                "returned": meta.get("returned", "?"),
                "fallback_used": meta.get("fallback_used", "?"),
                "mixed_space_skip_count": meta.get("mixed_space_skip_count", "?"),
                "strategy": meta.get("strategy", "?"),
                "evidence_strength": meta.get("evidence_strength", "?"),
                "latency_ms": ev.latency_ms,
                "created_at": str(ev.created_at)[:19] if ev.created_at else "?",
            })
        return result
    except Exception as exc:
        logger.warning("Cannot query retrieval events: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG diagnostics — inspect embedding, vector store, and ingestion state."
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--verbose", action="store_true", help="Detailed chunk-level breakdown")
    args = parser.parse_args()

    diagnostics: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "diagnostics_version": "1.0",
    }

    # 1. Active embedding config
    try:
        from app.core.config import settings
    except Exception as exc:
        logger.error("Cannot load app settings: %s", exc)
        print(json.dumps({"error": str(exc)}) if args.json else f"FATAL: {exc}")
        sys.exit(1)

    # P05-T03: include api_type in config diagnostics
    api_type_config = (settings.embedding_api_type or "").strip().lower()
    diagnostics["config"] = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "embedding_base_url": (settings.embedding_base_url or "").rstrip("/"),
        "embedding_api_type": api_type_config,
        "vector_store_provider": settings.vector_store_provider,
        "vector_store_collection": settings.vector_store_collection,
        "ingestion_pipeline": settings.ingestion_pipeline,
    }

    if not args.json:
        print("=" * 60)
        print("  RAG Diagnostics Report")
        print("=" * 60)
        print()
        print("  Configuration:")
        for key, val in diagnostics["config"].items():
            print(f"    {key}: {val}")

    # 2. Embedding health
    health = _check_embedding_health_local()
    diagnostics["embedding_health"] = health

    if not args.json:
        status = "✅ OK" if health.get("ok") else "❌ FAIL"
        print(f"\n  Embedding Health: {status}")
        print(f"    Provider: {health.get('provider', '?')}")
        # P05-T03: report api_type when present
        api_type_health = health.get("api_type", "")
        if api_type_health:
            print(f"    API type: {api_type_health}")
        if "model" in health:
            print(f"    Model:    {health['model']}")
        if "base_url" in health:
            print(f"    Base URL: {health['base_url']}")
        if "dimensions" in health:
            print(f"    Dims:     {health['dimensions']}")
        if "latency_ms" in health:
            print(f"    Latency:  {health['latency_ms']}ms")
        print(f"    Detail:   {health.get('detail', '?')}")

    # 3. Vector store health
    qdrant = _try_qdrant_health()
    diagnostics["vector_store"] = qdrant

    if not args.json:
        status = "✅ OK" if qdrant.get("ok") else "❌ FAIL"
        print(f"\n  Vector Store: {status}")
        print(f"    Active collection: {qdrant.get('active_collection', '?')}")
        active_info = qdrant.get("active_collection_info")
        if active_info:
            print(f"    Dimensions: {active_info['dimensions']}")
            print(f"    Points:     {active_info['points']}")
        print(f"    Detail:    {qdrant.get('detail', '?')}")
        if args.verbose and qdrant.get("collections"):
            print(f"    All collections: {', '.join(qdrant['collections'])}")

    # 4. Database check
    db_health = _try_db_health()
    diagnostics["database"] = db_health

    if not args.json:
        status = "✅ OK" if db_health.get("ok") else "❌ FAIL"
        print(f"\n  Database: {status}")

    # 5. Provider metadata counts (requires DB)
    db_session = None
    if db_health.get("ok"):
        try:
            from app.core.database import SessionLocal
            db_session = SessionLocal()
        except Exception as exc:
            logger.warning("Cannot create DB session: %s", exc)

    provider_counts = _query_provider_metadata_counts(db_session)
    diagnostics["provider_metadata_counts"] = provider_counts

    if not args.json:
        print(f"\n  Provider Metadata Distribution:")
        if provider_counts:
            for entry in provider_counts:
                print(f"    {entry['provider']}/{entry['model']} (dim={entry['dimensions']}): "
                      f"{entry['chunk_count']} chunks"
                      f"{'  [v=' + entry['embedding_version'] + ']' if entry['embedding_version'] else ''}")
        else:
            print("    (no data or database unavailable)")

    # 6. Documents needing reindex
    if db_session:
        needs_reindex = _find_documents_needing_reindex(db_session, provider_counts)
    else:
        needs_reindex = []
    diagnostics["documents_needing_reindex"] = needs_reindex

    if not args.json:
        print(f"\n  Documents Needing Reindex: {len(needs_reindex)}")
        for doc in needs_reindex:
            print(f"    [{doc['document_id']}] {doc['title']} (owner={doc['owner']})")
            print(f"      Reasons: {', '.join(doc['reasons'])}")

    # 7. Recent retrieval events
    if db_session:
        recent_events = _recent_retrieval_events(db_session)
    else:
        recent_events = []
    diagnostics["recent_retrieval_events"] = recent_events

    if not args.json:
        print(f"\n  Recent Retrieval Events (last {len(recent_events)}):")
        for ev in recent_events[:10]:
            fallback = " (FALLBACK)" if ev.get("fallback_used") else ""
            skip = f" skip={ev['mixed_space_skip_count']}" if ev.get("mixed_space_skip_count", 0) else ""
            print(f"    #{ev['id']} returned={ev['returned']}{fallback}{skip} "
                  f"[{ev['strategy']}] latency={ev['latency_ms']}ms")

    if db_session:
        try:
            db_session.close()
        except Exception:
            pass

    # 8. Diagnostics score
    issues = []
    if not health.get("ok"):
        issues.append(f"Embedding health failed: {health.get('detail', 'unknown')}")
    if not qdrant.get("ok"):
        issues.append(f"Vector store unreachable: {qdrant.get('detail', 'unknown')}")
    if not db_health.get("ok"):
        issues.append(f"Database unreachable: {db_health.get('detail', 'unknown')}")
    if needs_reindex:
        issues.append(f"{len(needs_reindex)} document(s) need reindexing")
    if provider_counts and len(provider_counts) > 1:
        issues.append(f"Multiple provider spaces detected ({len(provider_counts)} distinct)")

    diagnostics["issues"] = issues

    if not args.json:
        print(f"\n  {'=' * 60}")
        print(f"  Issues Found: {len(issues)}")
        for issue in issues:
            print(f"    ⚠ {issue}")
        if not issues:
            print("    ✅ All diagnostics passed — system looks healthy")
        print()
        print("  DIAGNOSE_RAG_DONE")
        print(f"  {'=' * 60}")
    else:
        print(json.dumps(diagnostics, indent=2, default=str))


if __name__ == "__main__":
    main()
