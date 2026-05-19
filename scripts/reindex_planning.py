"""
Reindex planning and execution script for Phase 4 collection migration.

Supports:
  - Listing documents with provider metadata (--list, --owner)
  - Validating Qdrant collection dimension compatibility (--validate)
  - Suggesting collection name for current provider/model (--suggest-collection)
  - Reindexing selected documents into the active provider's collection (--reindex)
  - Validating rollback safety (--rollback-check)

Usage:
    # List all documents with provider metadata
    python scripts/reindex_planning.py --list

    # List documents for a specific owner
    python scripts/reindex_planning.py --list --owner test_user

    # Validate Qdrant collection dimensions match current provider
    python scripts/reindex_planning.py --validate

    # Suggest recommended collection name for current provider/model
    python scripts/reindex_planning.py --suggest-collection

    # Reindex specific documents by ID
    python scripts/reindex_planning.py --reindex --doc-ids 1,2,3

    # Reindex all documents for an owner (max 50)
    python scripts/reindex_planning.py --reindex --owner test_user --limit 10

    # Check rollback readiness
    python scripts/reindex_planning.py --rollback-check

    # Dry-run reindex (report what would be done without executing)
    python scripts/reindex_planning.py --reindex --owner test_user --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

# Ensure the project root is on sys.path so app imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import get_logger
from app.crud import crud_knowledge
from app.models.knowledge_models import KnowledgeDocument, KnowledgeChunk
from app.services import vector_store
from app.services.embeddings.collection_naming import (
    suggest_collection_name,
    validate_collection_config,
)
from app.services.embeddings.factory import get_embedding_provider
from app.services.knowledge_service import reindex_document

logger = get_logger(__name__)

# ── ANSI helpers ──────────────────────────────────────────────────────────

def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if sys.stdout.isatty() else text

def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if sys.stdout.isatty() else text

def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if sys.stdout.isatty() else text

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if sys.stdout.isatty() else text

# ── Helpers ───────────────────────────────────────────────────────────────

def _get_current_provider_meta() -> dict:
    """Return dict of current provider metadata from the factory."""
    provider = get_embedding_provider()
    m = provider.meta
    return {
        "provider": m.provider,
        "model": m.model,
        "dimensions": m.dimensions,
        "version": m.version,
        "collection": settings.vector_store_collection,
        "vector_store": settings.vector_store_provider,
    }


def _get_chunk_provider_meta(chunk: KnowledgeChunk) -> dict:
    """Extract provider metadata from a chunk's metadata_json."""
    meta = (chunk.metadata_json or {}) if isinstance(chunk.metadata_json, dict) else {}
    return {
        "embedding_provider": meta.get("embedding_provider", "unknown"),
        "embedding_model": meta.get("embedding_model", "unknown"),
        "embedding_dimensions": meta.get("embedding_dimensions", 0),
        "embedding_version": meta.get("embedding_version", "unknown"),
        "parser_version": meta.get("parser_version", "unknown"),
        "chunker_version": meta.get("chunker_version", "unknown"),
    }


def _collection_info() -> dict | None:
    """Return Qdrant collection info if available, else None."""
    if not vector_store.is_external_vector_store_enabled():
        return None
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = vector_store._get_qdrant_client()
        info = client.get_collection(settings.vector_store_collection)
        return {
            "exists": True,
            "dimension": info.config.params.vectors.size,
            "points_count": info.points_count,
        }
    except (UnexpectedResponse, ValueError, RuntimeError) as exc:
        return {"exists": False, "error": str(exc)}


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> None:
    """List documents with provider metadata."""
    db = SessionLocal()
    try:
        owner_filter = args.owner
        if owner_filter:
            docs = crud_knowledge.list_documents(db, owner_filter, limit=args.limit or 50)
        else:
            # Fetch all owners by iterating
            all_docs: list[KnowledgeDocument] = (
                db.query(KnowledgeDocument)
                .filter(KnowledgeDocument.deleted_at.is_(None))
                .order_by(KnowledgeDocument.owner_username, KnowledgeDocument.created_at.desc())
                .limit(args.limit or 100)
                .all()
            )
            docs = all_docs

        if not docs:
            print(_yellow("No documents found."))
            return

        print(_bold(f"\n{'ID':>5} | {'Owner':<20} | {'Status':<10} | {'Chunks':>6} | {'Provider':<10} | {'Model':<25} | {'Dims':>5} | {'Collection'}"))
        print("-" * 120)

        for doc in docs:
            chunks = crud_knowledge.get_chunks_by_document(db, doc.id)
            provider_meta = _get_chunk_provider_meta(chunks[0]) if chunks else {}
            coll = getattr(doc.metadata_json, "get", lambda k, d=None: None)("vector_store_collection") if isinstance(doc.metadata_json, dict) else None
            print(
                f"{doc.id:>5} | {doc.owner_username:<20} | {doc.status:<10} | {len(chunks):>6} | "
                f"{provider_meta.get('embedding_provider', '-'):<10} | "
                f"{provider_meta.get('embedding_model', '-'):<25} | "
                f"{provider_meta.get('embedding_dimensions', '-'):>5} | "
                f"{coll or settings.vector_store_collection}"
            )

        print(f"\n{len(docs)} document(s) listed.")

    finally:
        db.close()


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate Qdrant collection dimensions and provider compatibility."""
    print(_bold("\n=== Provider Configuration ==="))
    meta = _get_current_provider_meta()
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # P03-T03: validate collection naming convention and print warnings
    provider = getattr(settings, "embedding_provider", "local")
    model = getattr(settings, "embedding_model", "unknown")
    collection_name = getattr(settings, "vector_store_collection", "knowledge_chunks")
    naming_warnings = validate_collection_config(provider, model, collection_name)
    if naming_warnings:
        print(_bold("\n=== Collection Naming Warnings ==="))
        for w in naming_warnings:
            print(_yellow(f"  ⚠ {w}"))

    print(_bold("\n=== Qdrant Collection ==="))
    coll = _collection_info()
    if coll is None:
        print(_yellow("  Qdrant is not enabled (VECTOR_STORE_PROVIDER is not 'qdrant')."))
        return

    if coll.get("exists"):
        dim = coll["dimension"]
        expected = meta["dimensions"]
        match = dim == expected
        status = _green("✓ MATCH") if match else _red("✗ MISMATCH")
        print(f"  Collection:       {settings.vector_store_collection}")
        print(f"  Vector dimension: {dim}")
        print(f"  Expected (meta):  {expected}")
        print(f"  Points count:     {coll.get('points_count', '?')}")
        print(f"  Status:           {status}")

        if not match:
            suggested = suggest_collection_name(provider, model)
            print(_red(
                f"\n  WARNING: Collection dimension ({dim}) does not match the current "
                f"provider's dimension ({expected}).\n"
                f"  Reindexing into this collection will FAIL at the dimension guard.\n"
                f"  Set VECTOR_STORE_COLLECTION to a dedicated collection like\n"
                f"  {suggested!r} for the new provider."
            ))
            sys.exit(1)
    else:
        print(_yellow(f"  Collection '{settings.vector_store_collection}' does not exist yet — will be created on first upsert."))
        print(f"  Expected dimension: {meta['dimensions']}")


def cmd_reindex(args: argparse.Namespace) -> None:
    """Reindex selected documents.

    Supports local, ollama, and api providers. For api providers, the reindex
    respects EMBEDDING_BATCH_SIZE for API rate limiting (handled internally by
    the GenericAPIProvider batch splitting). The old collection is never deleted,
    enabling rollback.
    """
    doc_ids: list[int] = []
    db = SessionLocal()
    try:
        # Resolve document IDs
        if args.doc_ids:
            doc_ids = [int(x.strip()) for x in args.doc_ids.split(",") if x.strip()]
        elif args.owner:
            docs = crud_knowledge.list_documents(db, args.owner, limit=args.limit or 50)
            doc_ids = [doc.id for doc in docs]
            if not doc_ids:
                print(_yellow(f"No documents found for owner '{args.owner}'."))
                return
        else:
            print(_red("Either --doc-ids or --owner is required for --reindex."))
            sys.exit(1)

        # Validate collection dimension before starting
        meta = _get_current_provider_meta()
        api_type = getattr(settings, "embedding_api_type", "") or ""
        batch_size = getattr(settings, "embedding_batch_size", 16)
        print(_bold(f"\n=== Reindex Plan ==="))
        print(f"  Provider:          {meta['provider']}")
        print(f"  Model:             {meta['model']}")
        print(f"  API type:          {api_type or '-'}")
        print(f"  Batch size:        {batch_size}")
        print(f"  Dimensions:        {meta['dimensions']}")
        print(f"  Collection:        {meta['collection']}")
        print(f"  Vector store:      {meta['vector_store']}")
        print(f"  Target documents:  {len(doc_ids)}")
        print(f"  Document IDs:      {doc_ids}")

        if vector_store.is_external_vector_store_enabled():
            coll = _collection_info()
            if coll and coll.get("exists"):
                dim = coll["dimension"]
                if dim != meta["dimensions"]:
                    print(_red(
                        f"\n  ✗ DIMENSION MISMATCH: Collection has dim={dim}, provider produces "
                        f"dim={meta['dimensions']}. Set VECTOR_STORE_COLLECTION to a dedicated "
                        f"collection before reindexing."
                    ))
                    sys.exit(1)
                print(_green(f"  ✓ Collection dimension ({dim}) matches provider."))
            else:
                print(_yellow(f"  ⚠ Collection '{meta['collection']}' will be created with dim={meta['dimensions']}."))

        # Dry run — report what would be done without calling API
        if args.dry_run:
            print(_yellow("\n  Dry-run mode — no documents will be modified."))
            print(f"  Provider:          {meta['provider']} ({meta['model']})")
            print(f"  Dimensions:        {meta['dimensions']}")
            if api_type:
                print(f"  API type:          {api_type}")
            print(f"  Batch size:        {batch_size}")
            for doc_id in doc_ids[:5]:
                print(f"    Would reindex doc_id={doc_id}")
            if len(doc_ids) > 5:
                print(f"    ... and {len(doc_ids) - 5} more")
            print("  Old collection preserved for rollback.")
            return

        # Confirm
        if not args.yes:
            confirm = input(f"\nReindex {len(doc_ids)} document(s)? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                return

        # Execute reindex
        print(_bold(f"\n=== Reindexing {len(doc_ids)} document(s) ==="))
        results: list[dict[str, Any]] = []
        success_count = 0
        fail_count = 0

        for idx, doc_id in enumerate(doc_ids):
            doc = crud_knowledge.get_document(db, doc_id)
            if not doc:
                print(f"  [{idx+1}/{len(doc_ids)}] SKIP doc_id={doc_id}: not found")
                results.append({
                    "doc_id": doc_id,
                    "status": "skipped",
                    "reason": "not_found",
                    "provider": meta["provider"],
                    "model": meta["model"],
                    "dimensions": meta["dimensions"],
                })
                continue

            print(f"  [{idx+1}/{len(doc_ids)}] Reindexing doc_id={doc_id} ('{doc.title[:50]}')... ", end="", flush=True)
            try:
                start = time.perf_counter()
                result = reindex_document(db, doc_id)
                elapsed = int((time.perf_counter() - start) * 1000)
                chunks = result.get("chunks_count", "?")
                print(_green(f"OK ({elapsed}ms, {chunks} chunks)"))
                results.append({
                    "doc_id": doc_id,
                    "title": doc.title,
                    "status": result.get("status", "ok"),
                    "chunks_count": chunks,
                    "pipeline": result.get("pipeline"),
                    "latency_ms": elapsed,
                    "provider": meta["provider"],
                    "model": meta["model"],
                    "dimensions": meta["dimensions"],
                })
                success_count += 1
            except Exception as e:
                print(_red(f"FAIL: {e}"))
                results.append({
                    "doc_id": doc_id,
                    "title": doc.title,
                    "status": "failed",
                    "error": str(e),
                    "provider": meta["provider"],
                    "model": meta["model"],
                    "dimensions": meta["dimensions"],
                })
                fail_count += 1

        # Summary
        print(_bold(f"\n=== Reindex Summary ==="))
        print(f"  Total:     {len(doc_ids)}")
        print(f"  Success:   {_green(str(success_count))}")
        print(f"  Failed:    {_red(str(fail_count)) if fail_count > 0 else '0'}")
        print(f"  Provider:  {meta['provider']}")
        print(f"  Model:     {meta['model']}")
        print(f"  API type:  {api_type or '-'}")
        print(f"  Dims:      {meta['dimensions']}")
        print(f"  Batch:     {batch_size}")
        print(f"  Coll:      {meta['collection']}")

        if fail_count > 0:
            for r in results:
                if r["status"] == "failed":
                    print(f"    doc_id={r['doc_id']}: {r.get('error', 'unknown error')}")
            sys.exit(1)

        print(_green("\n  ✓ Reindex complete. Old data in previous collection is preserved for rollback."))

    finally:
        db.close()


def cmd_rollback_check(args: argparse.Namespace) -> None:
    """Validate whether rollback to local-hash-v1 is safe.

    For API provider switches, this reports the previous provider's state,
    documents needing re-reindex, and recommended rollback settings.
    """
    print(_bold("\n=== Rollback Readiness Check ==="))

    # Check 1: Current provider
    meta = _get_current_provider_meta()
    provider_name = meta["provider"]
    model_name = meta["model"]
    api_type = getattr(settings, "embedding_api_type", "") or ""
    print(f"\n  Current provider:  {provider_name} ({model_name})")
    if api_type:
        print(f"  Current API type:  {api_type}")
    print(f"  Current collection: {meta['collection']}")
    print(f"  Current dimensions: {meta['dimensions']}")

    # Check 2: Old collection existence
    old_collection = "knowledge_chunks"
    if vector_store.is_external_vector_store_enabled():
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse
            client = vector_store._get_qdrant_client()
            info = client.get_collection(old_collection)
            old_dim = info.config.params.vectors.size
            print(f"\n  Old collection '{old_collection}': EXISTS (dim={old_dim}, {info.points_count} points)")
            if old_dim == 64:
                print(_green("  ✓ Old collection uses dim=64 (compatible with local-hash-v1)."))
            else:
                print(_yellow(f"  ⚠ Old collection dim={old_dim} — expected 64 for local-hash-v1."))
        except (UnexpectedResponse, ValueError):
            print(_yellow(f"\n  Old collection '{old_collection}' does not exist or is inaccessible."))
    else:
        print(_yellow(f"\n  Qdrant not enabled — no collection to validate."))

    # Check 3: Document provider provenance — detect any API provider chunks
    db = SessionLocal()
    try:
        indexed_docs = (
            db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.status == "indexed",
                KnowledgeDocument.deleted_at.is_(None),
            )
            .count()
        )
        print(f"\n  Indexed documents: {indexed_docs}")

        # Count chunks per embedding provider
        from sqlalchemy import text
        provider_counts: dict[str, int] = {}
        for prov in ("local", "ollama", "api", "openai", "cohere", "voyage", "huggingface"):
            cnt = (
                db.query(KnowledgeChunk)
                .filter(
                    KnowledgeChunk.metadata_json["embedding_provider"].as_string() == prov
                )
                .count()
            )
            if cnt:
                provider_counts[prov] = cnt

        if not provider_counts:
            print(_green("  ✓ No chunks found with recognized embedding providers."))
        else:
            print("  Chunks by provider:")
            for prov, cnt in sorted(provider_counts.items(), key=lambda x: -x[1]):
                is_current = (prov == provider_name) or (prov == "api" and provider_name == "api")
                marker = _green(" (active)") if is_current else ""
                print(f"    {prov}: {cnt}{marker}")

            api_related_chunks = sum(c for p, c in provider_counts.items() if p in ("api", "openai", "cohere", "voyage", "huggingface"))
            if api_related_chunks:
                print(_yellow(f"  ⚠ {api_related_chunks} chunk(s) from API provider(s) — these will need reindex after rollback."))
            else:
                print(_green("  ✓ No API provider chunks found — rollback requires no chunk changes."))

        # Check 4: Rollback settings
        print(_bold("\n  Rollback settings to restore:\n"))
        if provider_name == "api":
            print(f"    # --- Previous provider: {model_name} ---")
        print(f"    EMBEDDING_PROVIDER=local")
        print(f"    EMBEDDING_MODEL=local-hash-v1")
        print(f"    EMBEDDING_DIMENSIONS=64")
        print(f"    VECTOR_STORE_COLLECTION={old_collection}")
        if provider_name == "api":
            print(f"    # The following were in use before rollback:")
            print(f"    # EMBEDDING_BASE_URL={getattr(settings, 'embedding_base_url', '')}")
            if api_type:
                print(f"    # EMBEDDING_API_TYPE={api_type}")
    finally:
        db.close()

    print(_green("\n  Rollback check complete. See steps above for restoring local provider."))


# ── Suggest collection command ────────────────────────────────────────────


def cmd_suggest_collection(args: argparse.Namespace) -> None:
    """Suggest recommended collection name for current provider/model config."""
    meta = _get_current_provider_meta()
    provider = meta["provider"]
    model = meta["model"]
    dimensions = meta["dimensions"]
    current_collection = meta["collection"]
    suggested = suggest_collection_name(provider, model)

    print(_bold("\n=== Suggested Collection Name ==="))
    print(f"  Provider:            {provider}")
    print(f"  Model:               {model}")
    print(f"  Dimensions:          {dimensions}")
    print(f"  Current collection:  {current_collection}")
    print(f"  Suggested:           {_green(suggested)}")
    print()

    naming_warnings = validate_collection_config(provider, model, current_collection)
    if naming_warnings:
        print(_bold("=== Naming Warnings ==="))
        for w in naming_warnings:
            print(_yellow(f"  ⚠ {w}"))
        print()

    if current_collection != suggested:
        print(_yellow(
            f"  To switch to this suggested collection, set:\n"
            f"    VECTOR_STORE_COLLECTION={suggested}\n"
            f"  Then reindex documents into the new collection."
        ))


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reindex planning and execution for Phase 4 collection migration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("--list", aliases=["list"], help="List documents with provider metadata")
    p_list.add_argument("--owner", type=str, default=None, help="Filter by owner username")
    p_list.add_argument("--limit", type=int, default=100, help="Max documents to list")

    # validate
    p_validate = sub.add_parser("--validate", aliases=["validate"], help="Validate Qdrant collection dimensions and naming")
    # P03-T03: --validate now also checks collection naming convention

    # suggest-collection
    sub.add_parser("--suggest-collection", aliases=["suggest-collection"], help="Suggest collection name for current provider/model")

    # reindex
    p_reindex = sub.add_parser("--reindex", aliases=["reindex"], help="Reindex documents")
    p_reindex.add_argument("--doc-ids", type=str, default=None, help="Comma-separated document IDs")
    p_reindex.add_argument("--owner", type=str, default=None, help="Reindex all documents for this owner")
    p_reindex.add_argument("--limit", type=int, default=50, help="Max documents per owner")
    p_reindex.add_argument("--dry-run", action="store_true", help="Report what would be reindexed without executing")
    p_reindex.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    # rollback-check
    sub.add_parser("--rollback-check", aliases=["rollback-check"], help="Validate rollback readiness")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    command_map = {
        "--list": cmd_list,
        "list": cmd_list,
        "--validate": cmd_validate,
        "validate": cmd_validate,
        "--suggest-collection": cmd_suggest_collection,
        "suggest-collection": cmd_suggest_collection,
        "--reindex": cmd_reindex,
        "reindex": cmd_reindex,
        "--rollback-check": cmd_rollback_check,
        "rollback-check": cmd_rollback_check,
    }

    cmd = command_map.get(args.command)
    if cmd:
        cmd(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
