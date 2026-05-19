"""Read-only provider switch pre-flight checks for embedding migrations."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.database import SessionLocal
from app.crud import crud_knowledge
from app.models.knowledge_models import KnowledgeDocument
from app.services import vector_store
from app.services.embeddings.collection_naming import suggest_collection_name, validate_collection_config
from app.services.embeddings.factory import get_embedding_provider


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    guidance: str = ""


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _provider_meta() -> tuple[dict[str, Any], Any | None, Exception | None]:
    try:
        provider = get_embedding_provider()
        meta = provider.meta
        return {
            "provider": meta.provider,
            "model": meta.model,
            "dimensions": meta.dimensions,
            "version": meta.version,
            "api_type": getattr(settings, "embedding_api_type", "") or "",
            "collection": settings.vector_store_collection,
            "batch_size": settings.embedding_batch_size,
        }, provider, None
    except Exception as exc:  # pragma: no cover - exercised by CLI failure paths
        return {}, None, exc


def _collection_info(collection: str) -> dict[str, Any]:
    if not vector_store.is_external_vector_store_enabled():
        return {"enabled": False, "exists": False}
    try:
        client = vector_store._get_qdrant_client()
        info = client.get_collection(collection)
        return {
            "enabled": True,
            "exists": True,
            "dimension": info.config.params.vectors.size,
            "points_count": info.points_count,
        }
    except Exception as exc:
        return {"enabled": True, "exists": False, "error": str(exc)}


def run_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    meta, provider, provider_error = _provider_meta()

    if provider_error or provider is None:
        results.append(CheckResult(
            "provider configuration",
            False,
            f"Could not initialize provider: {provider_error}",
            "Check EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_BASE_URL, and provider-specific settings.",
        ))
        return results

    requires_key = bool(getattr(getattr(provider, "capabilities", None), "requires_api_key", False))
    api_key = getattr(settings, "embedding_api_key", "") or ""
    key_ok = (not requires_key) or bool(api_key)
    results.append(CheckResult(
        "API key requirement",
        key_ok,
        "API key is configured when required." if key_ok else "Provider requires an API key, but EMBEDDING_API_KEY is empty.",
        "Set EMBEDDING_API_KEY for the target provider before switching." if not key_ok else "",
    ))

    start = time.perf_counter()
    try:
        probe = provider.embed_query("provider switch preflight")
        latency_ms = int((time.perf_counter() - start) * 1000)
        detected_dims = len(probe.vector)
        target_dims = meta["dimensions"] or detected_dims
        reachable = detected_dims > 0
        results.append(CheckResult(
            "target provider reachable",
            reachable,
            f"Probe succeeded in {latency_ms}ms; detected dimensions={detected_dims}.",
            "Check network, base URL, credentials, model name, and provider rate limits." if not reachable else "",
        ))
        results.append(CheckResult(
            "target model available",
            True,
            f"Model {meta['model']!r} accepted an embedding request.",
        ))
        dims_ok = target_dims > 0 and (not meta["dimensions"] or meta["dimensions"] == detected_dims)
        results.append(CheckResult(
            "dimensions known/detected",
            dims_ok,
            f"Configured dimensions={meta['dimensions']}; detected dimensions={detected_dims}.",
            "Set EMBEDDING_DIMENSIONS to the provider's vector size or choose a matching model." if not dims_ok else "",
        ))
        meta["dimensions"] = target_dims
    except Exception as exc:
        safe = str(exc).replace(api_key, "***") if api_key else str(exc)
        results.append(CheckResult(
            "target provider reachable",
            False,
            f"Probe failed: {safe}",
            "Run scripts/embedding_provider_probe.py for detailed provider diagnostics.",
        ))
        results.append(CheckResult("target model available", False, "Model availability could not be confirmed because probe failed."))
        results.append(CheckResult("dimensions known/detected", bool(meta["dimensions"]), f"Configured dimensions={meta['dimensions']}."))

    collection = meta["collection"]
    coll = _collection_info(collection)
    if not coll.get("enabled"):
        results.append(CheckResult("target collection exists/can be created", True, "Qdrant is disabled; database vector fallback will be used."))
        results.append(CheckResult("no collection dimension conflict", True, "No Qdrant collection dimension to validate."))
    elif coll.get("exists"):
        results.append(CheckResult("target collection exists/can be created", True, f"Collection {collection!r} exists with {coll.get('points_count', '?')} points."))
        match = coll.get("dimension") == meta["dimensions"]
        results.append(CheckResult(
            "no collection dimension conflict",
            match,
            f"Collection dimension={coll.get('dimension')}; target dimensions={meta['dimensions']}.",
            f"Use VECTOR_STORE_COLLECTION={suggest_collection_name(meta['provider'], meta['model'])} or another empty collection for this embedding space." if not match else "",
        ))
    else:
        results.append(CheckResult("target collection exists/can be created", True, f"Collection {collection!r} does not exist and can be created on first write."))
        results.append(CheckResult("no collection dimension conflict", True, "No existing collection dimension conflict."))

    warnings = validate_collection_config(meta["provider"], meta["model"], collection)
    if warnings:
        results.append(CheckResult("collection naming convention", False, "; ".join(warnings), f"Recommended collection: {suggest_collection_name(meta['provider'], meta['model'])}"))
    else:
        results.append(CheckResult("collection naming convention", True, f"Collection {collection!r} matches provider/model naming guidance."))

    db = SessionLocal()
    try:
        doc_count = db.query(KnowledgeDocument).filter(KnowledgeDocument.deleted_at.is_(None)).count()
        indexed_count = db.query(KnowledgeDocument).filter(KnowledgeDocument.deleted_at.is_(None), KnowledgeDocument.status == "indexed").count()
        results.append(CheckResult("current documents and reindex scope", True, f"Documents={doc_count}; indexed documents estimated for reindex={indexed_count}; batch_size={meta['batch_size']}"))
    finally:
        db.close()

    return results


def main() -> None:
    print("=== Provider Switch Pre-flight (read-only) ===")
    print(f"Provider: {settings.embedding_provider}")
    print(f"Model:    {settings.embedding_model}")
    print(f"API type: {getattr(settings, 'embedding_api_type', '') or '-'}")
    print(f"Collection: {settings.vector_store_collection}\n")
    results = run_checks()
    for item in results:
        print(f"[{_status(item.ok)}] {item.name}: {item.detail}")
        if item.guidance:
            print(f"       guidance: {item.guidance}")
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\nPre-flight failed: {len(failed)} check(s) need attention.")
        sys.exit(1)
    print("\nPre-flight passed: provider switch prerequisites are satisfied.")


if __name__ == "__main__":
    main()
