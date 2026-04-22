from pathlib import Path
import io
import sys

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.deps import get_db
from app.core.database import Base
from app.main import app


LONG_TEXT = ("Alpha beta gamma delta epsilon zeta eta theta. " * 80).strip()
REFUND_TEXT = (
    "Product FAQ. Refund policy requests are reviewed within 5 business days. "
    "Customers can submit refund evidence through the support portal. " * 20
).strip()


def main() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "knowledge_user",
                "password": "StrongPass1!",
                "confirm_password": "StrongPass1!",
            },
        )
        assert register_response.status_code == 201, register_response.text
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        invalid_ingest_response = client.post(
            "/api/knowledge/ingest",
            json={"title": "Invalid doc", "source_type": "text"},
            headers=headers,
        )
        assert invalid_ingest_response.status_code == 400, invalid_ingest_response.text

        ingest_response = client.post(
            "/api/knowledge/documents/ingest",
            json={
                "title": "Product FAQ",
                "source_type": "text",
                "raw_text": LONG_TEXT,
                "metadata": {"category": "faq", "language": "en"},
            },
            headers=headers,
        )
        assert ingest_response.status_code == 201, ingest_response.text
        ingest_payload = ingest_response.json()
        assert ingest_payload["status"] == "indexed"
        assert ingest_payload["chunks_count"] >= 2
        text_doc_id = ingest_payload["document_id"]
        text_job_id = ingest_payload["job_id"]

        list_documents_response = client.get("/api/knowledge/documents", headers=headers)
        assert list_documents_response.status_code == 200, list_documents_response.text
        documents = list_documents_response.json()
        assert len(documents) == 1
        assert documents[0]["id"] == text_doc_id
        assert documents[0]["status"] == "indexed"
        assert documents[0]["metadata_json"]["category"] == "faq"

        document_detail_response = client.get(
            f"/api/knowledge/documents/{text_doc_id}",
            headers=headers,
        )
        assert document_detail_response.status_code == 200, document_detail_response.text
        assert document_detail_response.json()["title"] == "Product FAQ"

        chunks_response = client.get(
            f"/api/knowledge/documents/{text_doc_id}/chunks",
            headers=headers,
        )
        assert chunks_response.status_code == 200, chunks_response.text
        chunks = chunks_response.json()
        assert len(chunks) >= 2
        assert all(chunk["content"].strip() for chunk in chunks)
        assert all(chunk["embedding_model"] for chunk in chunks)
        assert all(chunk["vector_id"] for chunk in chunks)
        assert all(isinstance((chunk.get("metadata_json") or {}).get("embedding"), list) for chunk in chunks)

        search_response = client.post(
            "/api/knowledge/search",
            json={"query": "gamma epsilon faq", "top_k": 3},
            headers=headers,
        )
        assert search_response.status_code == 200, search_response.text
        search_payload = search_response.json()
        assert search_payload["returned"] >= 1
        assert search_payload["retrieval_id"] is not None
        assert search_payload["results"][0]["document_id"] == text_doc_id
        assert search_payload["results"][0]["score"] > 0

        filtered_search_response = client.post(
            "/api/knowledge/search",
            json={"query": "gamma epsilon faq", "top_k": 3, "document_id": text_doc_id},
            headers=headers,
        )
        assert filtered_search_response.status_code == 200, filtered_search_response.text
        filtered_search_payload = filtered_search_response.json()
        assert filtered_search_payload["strategy"] == "hybrid_rerank"
        assert filtered_search_payload["rewritten_query"] == "gamma epsilon faq"
        assert filtered_search_payload["fallback_used"] is False
        assert filtered_search_payload["results"][0]["rerank_score"] is not None
        assert filtered_search_payload["results"][0]["token_estimate"] is not None
        assert all(
            row["document_id"] == text_doc_id
            for row in filtered_search_payload["results"]
        )

        refund_ingest_response = client.post(
            "/api/knowledge/documents/ingest",
            json={
                "title": "Refund FAQ",
                "source_type": "text",
                "raw_text": REFUND_TEXT,
                "metadata": {"category": "policy", "language": "en"},
            },
            headers=headers,
        )
        assert refund_ingest_response.status_code == 201, refund_ingest_response.text
        refund_doc_id = refund_ingest_response.json()["document_id"]

        expanded_search_response = client.post(
            "/api/knowledge/search",
            json={"query": "Chính sách hoàn tiền xử lý bao lâu?", "top_k": 3, "document_id": refund_doc_id},
            headers=headers,
        )
        assert expanded_search_response.status_code == 200, expanded_search_response.text
        expanded_search_payload = expanded_search_response.json()
        assert expanded_search_payload["returned"] >= 1
        assert expanded_search_payload["strategy"] == "hybrid_rerank"
        assert expanded_search_payload["document_id"] == refund_doc_id
        assert expanded_search_payload["fallback_used"] is False
        assert expanded_search_payload["evidence_strength"] in {"grounded", "weak"}
        assert "refund" in (expanded_search_payload["rewritten_query"] or "").lower()
        assert len(expanded_search_payload["query_expansions"]) >= 1
        assert expanded_search_payload["results"][0]["document_id"] == refund_doc_id
        assert expanded_search_payload["results"][0]["lexical_score"] > 0
        assert expanded_search_payload["results"][0]["rerank_score"] is not None

        jobs_response = client.get(
            f"/api/knowledge/documents/{text_doc_id}/jobs",
            headers=headers,
        )
        assert jobs_response.status_code == 200, jobs_response.text
        jobs = jobs_response.json()
        assert len(jobs) == 1
        assert jobs[0]["id"] == text_job_id
        assert jobs[0]["status"] == "completed"

        job_detail_response = client.get(
            f"/api/knowledge/jobs/{text_job_id}",
            headers=headers,
        )
        assert job_detail_response.status_code == 200, job_detail_response.text
        assert job_detail_response.json()["status"] == "completed"

        duplicate_ingest_response = client.post(
            "/api/knowledge/ingest",
            json={
                "title": "Product FAQ Duplicate",
                "source_type": "text",
                "raw_text": LONG_TEXT,
            },
            headers=headers,
        )
        assert duplicate_ingest_response.status_code == 201, duplicate_ingest_response.text
        duplicate_payload = duplicate_ingest_response.json()
        assert duplicate_payload["document_id"] == text_doc_id
        assert duplicate_payload["job_id"] != text_job_id

        upload_response = client.post(
            "/api/knowledge/upload",
            headers=headers,
            files={
                "file": (
                    "release_notes.txt",
                    io.BytesIO(b"Release note one. Release note two. Release note three."),
                    "text/plain",
                )
            },
        )
        assert upload_response.status_code == 201, upload_response.text
        upload_payload = upload_response.json()
        uploaded_doc_id = upload_payload["document_id"]
        assert upload_payload["chunks_count"] >= 1

        unsupported_upload_response = client.post(
            "/api/knowledge/upload",
            headers=headers,
            files={
                "file": (
                    "archive.bin",
                    io.BytesIO(b"not supported"),
                    "application/octet-stream",
                )
            },
        )
        assert unsupported_upload_response.status_code == 400, unsupported_upload_response.text

        documents_after_upload = client.get("/api/knowledge/documents", headers=headers)
        assert documents_after_upload.status_code == 200, documents_after_upload.text
        assert len(documents_after_upload.json()) == 3

        reindex_response = client.post(
            f"/api/knowledge/documents/{text_doc_id}/reindex",
            headers=headers,
        )
        assert reindex_response.status_code == 200, reindex_response.text
        assert reindex_response.json()["status"] == "indexed"
        assert reindex_response.json()["chunks_count"] >= 2

        delete_response = client.delete(
            f"/api/knowledge/documents/{uploaded_doc_id}",
            headers=headers,
        )
        assert delete_response.status_code == 204, delete_response.text

        deleted_detail_response = client.get(
            f"/api/knowledge/documents/{uploaded_doc_id}",
            headers=headers,
        )
        assert deleted_detail_response.status_code == 404, deleted_detail_response.text

    app.dependency_overrides.clear()
    print("KNOWLEDGE_API_SMOKE_OK")


if __name__ == "__main__":
    main()

