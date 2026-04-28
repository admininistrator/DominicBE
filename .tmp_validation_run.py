import json
from app.api.endpoints import knowledge
print("KNOWLEDGE_ENDPOINT_IMPORT_OK")
from app.core.database import SessionLocal
from app.services.knowledge_service import backfill_documents_storage

db = SessionLocal()
try:
    summary = backfill_documents_storage(
        db,
        limit=1,
        write_object_artifacts=False,
        upsert_vectors=False,
        write_source_manifest=False,
    )
    compact = {
        key: summary.get(key)
        for key in ("selected_documents", "success_count", "error_count", "total_vector_points")
    }
    results = summary.get("results") or []
    if results:
        first = results[0]
        compact["first_result"] = {
            key: first.get(key)
            for key in (
                "document_id",
                "title",
                "owner_username",
                "object_storage",
                "vector_store",
                "vector_points",
                "source_manifest",
                "status",
                "error",
            )
        }
    print("BACKFILL_SUMMARY=" + json.dumps(compact, ensure_ascii=False, default=str))
finally:
    db.close()
