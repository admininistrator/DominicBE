import json
import app.main as main
from app.core.database import check_database_health
from app.services.object_storage import check_object_storage_health
from app.services.vector_store import check_vector_store_health


def route_summary(response):
    body = json.loads(response.body.decode("utf-8"))
    return {
        "status_code": response.status_code,
        "ok": body.get("ok"),
        "dependency": body.get("dependency"),
        "provider": body.get("provider"),
        "latency_ms": body.get("latency_ms"),
        "bucket_exists": body.get("bucket_exists"),
        "collection_exists": body.get("collection_exists"),
    }

probe_results = {
    "check_database_health": check_database_health(),
    "check_object_storage_health": check_object_storage_health(),
    "check_vector_store_health": check_vector_store_health(),
}

summary = {
    "probe_results": {
        name: {
            "ok": result.get("ok"),
            "dependency": result.get("dependency"),
            "provider": result.get("provider"),
            "latency_ms": result.get("latency_ms"),
            "bucket_exists": result.get("bucket_exists"),
            "collection_exists": result.get("collection_exists"),
        }
        for name, result in probe_results.items()
    },
    "route_results": {
        "health_postgres": route_summary(main.health_postgres()),
        "health_minio": route_summary(main.health_minio()),
        "health_qdrant": route_summary(main.health_qdrant()),
    },
}

print(json.dumps(summary, indent=2, sort_keys=True))
