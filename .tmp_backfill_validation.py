import json
from app.core.database import SessionLocal
from app.services.knowledge_service import backfill_documents_storage
import app.api.endpoints.knowledge

print('KNOWLEDGE_ENDPOINT_IMPORT_OK')

db = SessionLocal()
try:
    summary = backfill_documents_storage(
        db,
        limit=1,
        write_object_artifacts=False,
        upsert_vectors=False,
        write_source_manifest=False,
    )
    if hasattr(summary, 'dict'):
        data = summary.dict()
    elif isinstance(summary, dict):
        data = summary
    else:
        data = {
            k: getattr(summary, k)
            for k in dir(summary)
            if not k.startswith('_') and not callable(getattr(summary, k))
        }
    preferred = [
        'processed',
        'updated',
        'skipped',
        'errors',
        'limit',
        'documents_seen',
        'documents_processed',
        'documents_updated',
        'artifacts_written',
        'vectors_upserted',
        'source_manifests_written',
    ]
    key_only = {k: data.get(k) for k in preferred if k in data}
    if not key_only:
        key_only = data
    print('BACKFILL_SUMMARY=' + json.dumps(key_only, default=str, sort_keys=True))
finally:
    db.close()
