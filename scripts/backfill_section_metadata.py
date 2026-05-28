from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
import app.models.chat_models  # noqa: F401
from app.models.knowledge_models import KnowledgeChunk, KnowledgeDocument
from app.services.knowledge_service import reindex_document


SPAN_METADATA_KEYS = ("char_start", "char_end")
PAGE_METADATA_KEYS = ("page_number", "page_range")


@dataclass(frozen=True)
class SectionMetadataCandidate:
    document_id: int
    owner_username: str
    title: str
    status: str
    source_type: str | None
    mime_type: str | None
    has_raw_text: bool
    total_chunks: int
    chunks_with_section_key: int
    chunks_with_span_metadata: int
    chunks_with_page_metadata: int
    page_metadata_expected: bool
    needs_reindex: bool
    reason: str


def _metadata(row: KnowledgeChunk) -> dict[str, Any]:
    return row.metadata_json if isinstance(row.metadata_json, dict) else {}


def _has_any(meta: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(meta.get(key) is not None for key in keys)


def _page_metadata_expected(doc: KnowledgeDocument) -> bool:
    mime_type = (doc.mime_type or "").lower()
    source_uri = (doc.source_uri or "").lower()
    title = (doc.title or "").lower()
    return "pdf" in mime_type or source_uri.endswith(".pdf") or title.endswith(".pdf")


def _candidate_for_document(db: Session, doc: KnowledgeDocument, *, force: bool) -> SectionMetadataCandidate:
    chunk_rows = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.document_id == doc.id)
        .order_by(KnowledgeChunk.chunk_index.asc())
        .all()
    )
    total_chunks = len(chunk_rows)
    chunks_with_section_key = 0
    chunks_with_span_metadata = 0
    chunks_with_page_metadata = 0
    for row in chunk_rows:
        meta = _metadata(row)
        if meta.get("section_key"):
            chunks_with_section_key += 1
        if _has_any(meta, SPAN_METADATA_KEYS):
            chunks_with_span_metadata += 1
        if _has_any(meta, PAGE_METADATA_KEYS):
            chunks_with_page_metadata += 1

    page_expected = _page_metadata_expected(doc)
    missing_section = total_chunks > 0 and chunks_with_section_key == 0
    missing_span = total_chunks > 0 and chunks_with_span_metadata == 0
    missing_page = page_expected and total_chunks > 0 and chunks_with_page_metadata == 0
    needs_reindex = force or missing_section or missing_span or missing_page or total_chunks == 0

    if force:
        reason = "forced"
    elif total_chunks == 0:
        reason = "no_chunks"
    elif missing_page:
        reason = "missing_page_metadata"
    elif missing_section and missing_span:
        reason = "missing_section_and_span_metadata"
    elif missing_section:
        reason = "missing_section_metadata"
    elif missing_span:
        reason = "missing_span_metadata"
    else:
        reason = "metadata_present"

    return SectionMetadataCandidate(
        document_id=int(doc.id),
        owner_username=doc.owner_username,
        title=doc.title,
        status=doc.status,
        source_type=doc.source_type,
        mime_type=doc.mime_type,
        has_raw_text=bool(doc.raw_text),
        total_chunks=total_chunks,
        chunks_with_section_key=chunks_with_section_key,
        chunks_with_span_metadata=chunks_with_span_metadata,
        chunks_with_page_metadata=chunks_with_page_metadata,
        page_metadata_expected=page_expected,
        needs_reindex=needs_reindex,
        reason=reason,
    )


def collect_reindex_candidates(
    db: Session,
    *,
    document_ids: list[int] | None = None,
    owner_username: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> list[SectionMetadataCandidate]:
    query = (
        db.query(KnowledgeDocument)
        .filter(
            KnowledgeDocument.deleted_at.is_(None),
            KnowledgeDocument.status == "indexed",
        )
        .order_by(KnowledgeDocument.id.asc())
    )
    if document_ids:
        query = query.filter(KnowledgeDocument.id.in_(document_ids))
    if owner_username:
        query = query.filter(KnowledgeDocument.owner_username == owner_username)
    if limit is not None and limit > 0:
        query = query.limit(limit)
    return [_candidate_for_document(db, doc, force=force) for doc in query.all()]


def run_backfill(
    db: Session,
    *,
    document_ids: list[int] | None = None,
    owner_username: str | None = None,
    limit: int | None = None,
    apply: bool = False,
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    candidates = collect_reindex_candidates(
        db,
        document_ids=document_ids,
        owner_username=owner_username,
        limit=limit,
        force=force,
    )
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        item = asdict(candidate)
        if not candidate.needs_reindex:
            item.update(status="skipped_up_to_date", chunks_after=None, error=None)
            results.append(item)
            continue
        if not candidate.has_raw_text:
            item.update(
                status="skipped_missing_raw_text",
                chunks_after=None,
                error="Document has no raw_text; re-upload or restore source text before re-indexing.",
            )
            results.append(item)
            continue
        if not apply:
            item.update(status="dry_run", chunks_after=None, error=None)
            results.append(item)
            continue
        try:
            reindex_result = reindex_document(db, candidate.document_id)
            item.update(
                status="reindexed",
                chunks_after=reindex_result.get("chunks_count"),
                job_id=reindex_result.get("job_id"),
                error=None,
            )
        except Exception as exc:
            item.update(status="error", chunks_after=None, error=exc.__class__.__name__)
            if fail_fast:
                results.append(item)
                raise
        results.append(item)

    return {
        "mode": "apply" if apply else "dry_run",
        "selected_documents": len(candidates),
        "needs_reindex_count": sum(1 for item in results if item["needs_reindex"]),
        "reindexed_count": sum(1 for item in results if item["status"] == "reindexed"),
        "skipped_count": sum(1 for item in results if item["status"].startswith("skipped")),
        "error_count": sum(1 for item in results if item["status"] == "error"),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or apply re-indexing for old indexed documents so newly generated chunks "
            "receive section/span/page metadata. Default mode is safe dry-run."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview candidate documents without modifying chunks. This is the default.")
    mode.add_argument("--apply", action="store_true", help="Actually re-index selected documents that need metadata.")
    parser.add_argument("--document-id", action="append", type=int, default=[], help="Specific document id to inspect/re-index. Can be repeated.")
    parser.add_argument("--owner", default=None, help="Limit to one owner_username.")
    parser.add_argument("--limit", type=int, default=None, help="Batch size / maximum number of indexed documents to inspect.")
    parser.add_argument("--force", action="store_true", help="Re-index even when metadata appears present.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first apply-mode error.")
    return parser


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"[info] mode={summary['mode']} selected_documents={summary['selected_documents']} needs_reindex={summary['needs_reindex_count']}")
    for item in summary["results"]:
        print(
            "[{status}] doc_id={document_id} owner={owner_username} title={title!r} "
            "reason={reason} chunks={total_chunks} section_chunks={chunks_with_section_key} "
            "span_chunks={chunks_with_span_metadata} page_chunks={chunks_with_page_metadata} "
            "raw_text={has_raw_text}".format(**item)
        )
        if item.get("error"):
            print(f"  error={item['error']}")
    print(
        f"[summary] reindexed={summary['reindexed_count']} skipped={summary['skipped_count']} errors={summary['error_count']}"
    )


def main() -> None:
    args = build_parser().parse_args()
    db = SessionLocal()
    try:
        summary = run_backfill(
            db,
            document_ids=args.document_id,
            owner_username=args.owner,
            limit=args.limit,
            apply=bool(args.apply),
            force=args.force,
            fail_fast=args.fail_fast,
        )
        _print_summary(summary)
        if not args.apply:
            print("[info] Dry-run only. Re-run with --apply to re-index selected documents.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
