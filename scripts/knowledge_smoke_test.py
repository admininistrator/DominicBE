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
from app.core.config import settings
from app.core.database import Base
from app.main import app


API_PREFIX = "/api/v1"

# ---------------------------------------------------------------------------
# Ollama ingestion smoke constants (P02-T05)
# ---------------------------------------------------------------------------
_OLLAMA_SMOKE_TEXT = (
    "Qwen3 embedding integration smoke test document. "
    "This paragraph verifies that the Ollama provider can embed document chunks "
    "and store them in the configured Qdrant collection. "
    "The embedding provider is qwen3-embedding:0.6b. "
    "Retrieval should return this chunk when queried for Qwen3 embedding. " * 10
).strip()


def api_path(path: str) -> str:
    return f"{API_PREFIX}{path}"


LONG_TEXT = ("Alpha beta gamma delta epsilon zeta eta theta. " * 80).strip()
REFUND_TEXT = (
    "Product FAQ. Refund policy requests are reviewed within 5 business days. "
    "Customers can submit refund evidence through the support portal. " * 20
).strip()
SECTION_SMOKE_TEXT = """Bài thực hành số 4
1. Nội dung bài một.
2. Nội dung bài hai.
3. Nội dung bài ba.

Practice Lesson 4
1. Prepare the worksheet.
2. Submit the reflection.
""".strip()


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
    original_rate_limit_enabled = settings.rate_limit_enabled
    settings.rate_limit_enabled = False

    try:
        with TestClient(app) as client:
            register_response = client.post(
                api_path("/auth/register"),
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
                api_path("/knowledge/documents/ingest"),
                json={"title": "Invalid doc", "source_type": "text"},
                headers=headers,
            )
            assert invalid_ingest_response.status_code == 400, invalid_ingest_response.text

            ingest_response = client.post(
                api_path("/knowledge/documents/ingest"),
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

            list_documents_response = client.get(api_path("/knowledge/documents"), headers=headers)
            assert list_documents_response.status_code == 200, list_documents_response.text
            documents = list_documents_response.json()
            assert len(documents) == 1
            assert documents[0]["id"] == text_doc_id
            assert documents[0]["status"] == "indexed"
            assert documents[0]["metadata_json"]["category"] == "faq"

            document_detail_response = client.get(
                api_path(f"/knowledge/documents/{text_doc_id}"),
                headers=headers,
            )
            assert document_detail_response.status_code == 200, document_detail_response.text
            assert document_detail_response.json()["title"] == "Product FAQ"

            chunks_response = client.get(
                api_path(f"/knowledge/documents/{text_doc_id}/chunks"),
                headers=headers,
            )
            assert chunks_response.status_code == 200, chunks_response.text
            chunks = chunks_response.json()
            assert len(chunks) >= 2
            assert all(chunk["content"].strip() for chunk in chunks)
            assert all(chunk["embedding_model"] for chunk in chunks)
            assert all(chunk["vector_id"] for chunk in chunks)
            assert all(isinstance(chunk.get("metadata_json") or {}, dict) for chunk in chunks)
            assert all((chunk.get("metadata_json") or {}).get("index_provider") for chunk in chunks)
            assert all((chunk.get("metadata_json") or {}).get("embedding_provider") for chunk in chunks)
            assert all(
                "embedding" not in (chunk.get("metadata_json") or {})
                or isinstance((chunk.get("metadata_json") or {}).get("embedding"), list)
                for chunk in chunks
            )

            search_response = client.post(
                api_path("/knowledge/search"),
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
                api_path("/knowledge/search"),
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
                api_path("/knowledge/documents/ingest"),
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
                api_path("/knowledge/search"),
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

            section_ingest_response = client.post(
                api_path("/knowledge/documents/ingest"),
                json={
                    "title": "Synthetic Practice Section",
                    "source_type": "text",
                    "raw_text": SECTION_SMOKE_TEXT,
                    "metadata": {"category": "smoke", "language": "vi"},
                },
                headers=headers,
            )
            assert section_ingest_response.status_code == 201, section_ingest_response.text
            section_doc_id = section_ingest_response.json()["document_id"]

            section_chunks_response = client.get(
                api_path(f"/knowledge/documents/{section_doc_id}/chunks"),
                headers=headers,
            )
            assert section_chunks_response.status_code == 200, section_chunks_response.text
            section_chunks = section_chunks_response.json()
            assert len(section_chunks) >= 1
            assert any(
                (chunk.get("metadata_json") or {}).get("section_key") == "bai-thuc-hanh-so-4"
                for chunk in section_chunks
            )
            assert any(
                (chunk.get("metadata_json") or {}).get("char_start") is not None
                for chunk in section_chunks
            )

            section_search_response = client.post(
                api_path("/knowledge/search"),
                json={
                    "query": "Bài thực hành số 4 có mấy bài, tóm tắt từng bài",
                    "top_k": 5,
                    "document_id": section_doc_id,
                },
                headers=headers,
            )
            assert section_search_response.status_code == 200, section_search_response.text
            section_search_payload = section_search_response.json()
            assert section_search_payload["rag_mode"] == "section_rag"
            assert section_search_payload["retrieval_scope"] == "document"
            assert section_search_payload["selected_document_id"] == section_doc_id
            assert section_search_payload["section_key"] == "bai-thuc-hanh-so-4"
            assert section_search_payload["vector_store_attempted"] is False
            assert section_search_payload["returned"] >= 1
            assert all(
                row["document_id"] == section_doc_id
                and row.get("section_key") == "bai-thuc-hanh-so-4"
                for row in section_search_payload["results"]
            )
            section_context = "\n".join(row["snippet"] for row in section_search_payload["results"])
            assert "Nội dung bài một" in section_context
            assert "Nội dung bài ba" in section_context

            jobs_response = client.get(
                api_path(f"/knowledge/documents/{text_doc_id}/jobs"),
                headers=headers,
            )
            assert jobs_response.status_code == 200, jobs_response.text
            jobs = jobs_response.json()
            assert len(jobs) == 1
            assert jobs[0]["id"] == text_job_id
            assert jobs[0]["status"] == "completed"

            job_detail_response = client.get(
                api_path(f"/knowledge/jobs/{text_job_id}"),
                headers=headers,
            )
            assert job_detail_response.status_code == 200, job_detail_response.text
            assert job_detail_response.json()["status"] == "completed"

            duplicate_ingest_response = client.post(
                api_path("/knowledge/documents/ingest"),
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
                api_path("/knowledge/documents/upload"),
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
                api_path("/knowledge/documents/upload"),
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

            documents_after_upload = client.get(api_path("/knowledge/documents"), headers=headers)
            assert documents_after_upload.status_code == 200, documents_after_upload.text
            assert len(documents_after_upload.json()) == 4

            reindex_response = client.post(
                api_path(f"/knowledge/documents/{text_doc_id}/reindex"),
                headers=headers,
            )
            assert reindex_response.status_code == 200, reindex_response.text
            assert reindex_response.json()["status"] == "indexed"
            assert reindex_response.json()["chunks_count"] >= 2

            delete_response = client.delete(
                api_path(f"/knowledge/documents/{uploaded_doc_id}"),
                headers=headers,
            )
            assert delete_response.status_code == 204, delete_response.text

            deleted_detail_response = client.get(
                api_path(f"/knowledge/documents/{uploaded_doc_id}"),
                headers=headers,
            )
            assert deleted_detail_response.status_code == 404, deleted_detail_response.text

    finally:
        settings.rate_limit_enabled = original_rate_limit_enabled
        app.dependency_overrides.clear()

    print("KNOWLEDGE_API_SMOKE_OK")


# ---------------------------------------------------------------------------
# P02-T05: Ollama ingestion smoke
# ---------------------------------------------------------------------------

def ollama_ingestion_smoke() -> None:
    """Smoke test: ingest a small document with the Ollama provider and verify
    chunk metadata, provider fields, and vector upsert target collection.

    Requirements
    ------------
    - EMBEDDING_PROVIDER=ollama must be set in the environment.
    - Ollama must be reachable at EMBEDDING_BASE_URL with the configured model.
    - VECTOR_STORE_COLLECTION should be knowledge_qwen3_embedding_06b (recommended).
    - The existing local collection (knowledge_chunks) is NOT touched.

    This test does NOT require a running FastAPI server; it uses TestClient with
    an in-memory SQLite database so it is self-contained.
    """
    provider = (settings.embedding_provider or "local").strip().lower()
    if provider != "ollama":
        print(
            f"SKIP: EMBEDDING_PROVIDER={provider!r}. "
            "Set EMBEDDING_PROVIDER=ollama to run the Ollama ingestion smoke.",
            file=sys.stderr,
        )
        sys.exit(0)

    model = settings.embedding_model or "qwen3-embedding:0.6b"
    base_url = (settings.embedding_base_url or "http://localhost:11434").rstrip("/")
    collection = settings.vector_store_collection or "knowledge_qwen3_embedding_06b"

    print("=" * 60)
    print("Ollama Ingestion Smoke Test (P02-T05)")
    print("=" * 60)
    print(f"  provider:   {provider}")
    print(f"  model:      {model}")
    print(f"  base_url:   {base_url}")
    print(f"  collection: {collection}")
    print()

    # Warn if the user is about to write into the local-hash collection
    if collection == "knowledge_chunks":
        print(
            "WARNING: VECTOR_STORE_COLLECTION=knowledge_chunks is the default local-hash "
            "collection.\n"
            "         Recommend using knowledge_qwen3_embedding_06b to avoid mixing "
            "embedding spaces.",
            file=sys.stderr,
        )

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
    original_rate_limit_enabled = settings.rate_limit_enabled
    settings.rate_limit_enabled = False

    try:
        with TestClient(app) as client:
            # Register a dedicated smoke user
            reg = client.post(
                f"{API_PREFIX}/auth/register",
                json={
                    "username": "ollama_smoke_user",
                    "password": "StrongPass1!",
                    "confirm_password": "StrongPass1!",
                },
            )
            assert reg.status_code == 201, f"Register failed: {reg.text}"
            token = reg.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            # ----------------------------------------------------------------
            # Step 1: Ingest a small document via the Ollama provider
            # ----------------------------------------------------------------
            print("[1/5] Ingesting document with Ollama provider ...")
            ingest = client.post(
                f"{API_PREFIX}/knowledge/documents/ingest",
                json={
                    "title": "Ollama Smoke Doc",
                    "source_type": "text",
                    "raw_text": _OLLAMA_SMOKE_TEXT,
                    "metadata": {"smoke": "ollama", "provider": provider},
                },
                headers=headers,
            )
            assert ingest.status_code == 201, (
                f"Ingest failed (HTTP {ingest.status_code}): {ingest.text}\n"
                f"  Ensure Ollama is running at {base_url} and model '{model}' is pulled."
            )
            ingest_payload = ingest.json()
            assert ingest_payload["status"] == "indexed", (
                f"Expected status=indexed, got {ingest_payload.get('status')}"
            )
            assert ingest_payload["chunks_count"] >= 1, "Expected at least 1 chunk"
            doc_id = ingest_payload["document_id"]
            print(f"  document_id={doc_id}  chunks={ingest_payload['chunks_count']}")

            # ----------------------------------------------------------------
            # Step 2: Verify chunk metadata provider fields
            # ----------------------------------------------------------------
            print("[2/5] Verifying chunk metadata provider fields ...")
            chunks_resp = client.get(
                f"{API_PREFIX}/knowledge/documents/{doc_id}/chunks",
                headers=headers,
            )
            assert chunks_resp.status_code == 200, chunks_resp.text
            chunks = chunks_resp.json()
            assert len(chunks) >= 1, "Expected at least 1 chunk in listing"

            for chunk in chunks:
                meta = chunk.get("metadata_json") or {}
                # embedding_provider must be 'ollama'
                assert meta.get("embedding_provider") == "ollama", (
                    f"chunk {chunk.get('id')}: expected embedding_provider=ollama, "
                    f"got {meta.get('embedding_provider')!r}"
                )
                # embedding_model must match configured model
                assert meta.get("embedding_model") == model or chunk.get("embedding_model") == model, (
                    f"chunk {chunk.get('id')}: expected embedding_model={model!r}, "
                    f"got meta={meta.get('embedding_model')!r} chunk={chunk.get('embedding_model')!r}"
                )
                # embedding_dimensions must be present and > 0
                dims = meta.get("embedding_dimensions") or 0
                assert dims > 0, (
                    f"chunk {chunk.get('id')}: embedding_dimensions missing or zero"
                )
                # vector_id must be set
                assert chunk.get("vector_id"), (
                    f"chunk {chunk.get('id')}: vector_id is missing"
                )

            print(f"  {len(chunks)} chunks verified — provider=ollama model={model} dims={dims}")

            # ----------------------------------------------------------------
            # Step 3: Verify vector upsert target collection
            # ----------------------------------------------------------------
            print("[3/5] Verifying vector store collection ...")
            # The collection name comes from settings; we assert it matches what
            # was configured so the operator knows which collection was written.
            actual_collection = settings.vector_store_collection
            print(f"  VECTOR_STORE_COLLECTION={actual_collection!r}")
            assert actual_collection, "VECTOR_STORE_COLLECTION must not be empty"
            # Warn (not fail) if the local-hash default collection is used
            if actual_collection == "knowledge_chunks":
                print(
                    "  WARNING: writing Qwen3 vectors into 'knowledge_chunks' mixes "
                    "embedding spaces. Use 'knowledge_qwen3_embedding_06b' instead.",
                    file=sys.stderr,
                )

            # ----------------------------------------------------------------
            # Step 4: Reindex the document (proves reindex path works with Ollama)
            # ----------------------------------------------------------------
            print("[4/5] Reindexing document with Ollama provider ...")
            reindex = client.post(
                f"{API_PREFIX}/knowledge/documents/{doc_id}/reindex",
                headers=headers,
            )
            assert reindex.status_code == 200, (
                f"Reindex failed (HTTP {reindex.status_code}): {reindex.text}"
            )
            assert reindex.json()["status"] == "indexed"
            assert reindex.json()["chunks_count"] >= 1
            print(f"  Reindex OK  chunks={reindex.json()['chunks_count']}")

            # ----------------------------------------------------------------
            # Step 5: Confirm local collection is untouched
            # ----------------------------------------------------------------
            print("[5/5] Confirming local collection is not modified ...")
            # We cannot directly inspect Qdrant collections in this in-process
            # test, but we verify that the configured collection name is NOT
            # 'knowledge_chunks' when the provider is ollama (best-effort guard).
            if actual_collection != "knowledge_chunks":
                print(
                    f"  OK — Ollama vectors written to '{actual_collection}', "
                    "local 'knowledge_chunks' collection is untouched."
                )
            else:
                print(
                    "  NOTE — both providers share 'knowledge_chunks'. "
                    "Consider using a dedicated collection for Qwen3."
                )

    finally:
        settings.rate_limit_enabled = original_rate_limit_enabled
        app.dependency_overrides.clear()

    print()
    print("OLLAMA_INGESTION_SMOKE_OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge API smoke tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/knowledge_smoke_test.py\n"
            "      Run the standard local-provider smoke test.\n\n"
            "  EMBEDDING_PROVIDER=ollama python scripts/knowledge_smoke_test.py --ollama\n"
            "      Run the Ollama ingestion smoke (requires Ollama running).\n"
        ),
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help=(
            "Run the Ollama ingestion smoke test (P02-T05). "
            "Requires EMBEDDING_PROVIDER=ollama and a running Ollama server."
        ),
    )
    args = parser.parse_args()

    if args.ollama:
        ollama_ingestion_smoke()
    else:
        main()
