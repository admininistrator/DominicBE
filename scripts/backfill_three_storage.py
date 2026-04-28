from __future__ import annotations

import argparse

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
import app.models.chat_models  # noqa: F401
from app.services.knowledge_service import backfill_documents_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill existing knowledge documents into object storage and vector store.",
    )
    parser.add_argument("--document-id", action="append", type=int, default=[], help="Specific document id to backfill. Can be repeated.")
    parser.add_argument("--owner", default=None, help="Limit to one owner_username.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to process.")
    parser.add_argument("--skip-object-storage", action="store_true", help="Do not write normalized/source-status artifacts.")
    parser.add_argument("--skip-vector-store", action="store_true", help="Do not backfill vectors into Qdrant.")
    parser.add_argument("--skip-source-manifest", action="store_true", help="Do not write a source-status manifest for legacy docs without original bytes.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first document error.")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    db = SessionLocal()
    try:
        summary = backfill_documents_storage(
            db,
            document_ids=args.document_id,
            owner_username=args.owner,
            limit=args.limit,
            write_object_artifacts=not args.skip_object_storage,
            upsert_vectors=not args.skip_vector_store,
            write_source_manifest=not args.skip_source_manifest,
            fail_fast=args.fail_fast,
        )
        print(f"[info] documents_selected={summary['selected_documents']}")
        for result in summary["results"]:
            if result["status"] == "ok":
                print(
                    "[ok] doc_id={document_id} owner={owner_username} object_storage={object_storage} "
                    "vector_store={vector_store} vector_points={vector_points} source_manifest={source_manifest}".format(**result)
                )
            else:
                print(f"[error] doc_id={result['document_id']} owner={result['owner_username']} error={result['error']}")
        print(
            f"[summary] success={summary['success_count']} errors={summary['error_count']} "
            f"vector_points={summary['total_vector_points']}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()