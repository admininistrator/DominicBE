from pathlib import Path
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
from app.services import chat_service


API_PREFIX = "/api/v1"


def api_path(path: str) -> str:
    return f"{API_PREFIX}{path}"


LONG_TEXT = (
    "Dominic Product FAQ. Refund requests are reviewed within 5 business days. "
    "Customers can submit refund evidence through the support portal. " * 40
).strip()

SECTION_TEXT = """Bài thực hành số 4
1. Nội dung bài một.
2. Nội dung bài hai.
3. Nội dung bài ba.
""".strip()

def _fake_complete(*, messages, system=None, max_tokens=1024, **kwargs):
    assert system
    assert "Evidence for this turn" in system
    assert messages
    assert max_tokens >= 1
    if "Bài thực hành số 4" in system:
        assert "Nội dung bài một" in system
        assert "Nội dung bài ba" in system
        return {
            "text": "Bài thực hành số 4 có 3 bài: bài một, bài hai và bài ba. [Source 1]",
            "input_tokens": 120,
            "output_tokens": 48,
            "model": "test/fake-model",
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
    assert "Product FAQ" in system
    assert "refund requests are reviewed within 5 business days" in system.lower()
    return {
        "text": "Theo Product FAQ, yêu cầu hoàn tiền được xem xét trong vòng 5 ngày làm việc. [Source 1]",
        "input_tokens": 120,
        "output_tokens": 48,
        "model": "test/fake-model",
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


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

    original_complete = chat_service.llm_provider.complete
    app.dependency_overrides[get_db] = override_get_db
    chat_service.llm_provider.complete = _fake_complete
    original_rate_limit_enabled = settings.rate_limit_enabled
    settings.rate_limit_enabled = False

    try:
        with TestClient(app) as client:
            register_response = client.post(
                api_path("/auth/register"),
                json={
                    "username": "rag_user",
                    "password": "StrongPass1!",
                    "confirm_password": "StrongPass1!",
                },
            )
            assert register_response.status_code == 201, register_response.text
            token = register_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            ingest_response = client.post(
                api_path("/knowledge/documents/ingest"),
                json={
                    "title": "Product FAQ",
                    "source_type": "text",
                    "raw_text": LONG_TEXT,
                },
                headers=headers,
            )
            assert ingest_response.status_code == 201, ingest_response.text
            document_id = ingest_response.json()["document_id"]

            session_response = client.post(
                api_path("/chat/sessions"),
                json={"title": "Grounded chat"},
                headers=headers,
            )
            assert session_response.status_code == 200, session_response.text
            session_id = session_response.json()["id"]

            chat_response = client.post(
                api_path("/chat/"),
                json={
                    "session_id": session_id,
                    "message": "Chính sách hoàn tiền xử lý trong bao lâu?",
                    "knowledge_document_id": document_id,
                },
                headers=headers,
            )
            assert chat_response.status_code == 200, chat_response.text
            chat_payload = chat_response.json()
            assert chat_payload["reply"]
            assert chat_payload["request_id"]
            assert chat_payload["retrieval"]["used"] is True
            assert chat_payload["retrieval"]["returned"] >= 1
            assert chat_payload["retrieval"]["document_id"] == document_id
            assert chat_payload["retrieval"]["strategy"] == "hybrid_rerank"
            assert chat_payload["retrieval"]["fallback_used"] is False
            assert "refund" in (chat_payload["retrieval"]["rewritten_query"] or "").lower()
            assert len(chat_payload["retrieval"]["query_expansions"]) >= 1
            assert chat_payload["retrieval"]["evidence_strength"] in {"grounded", "weak"}
            assert chat_payload["retrieval"]["answer_policy"] == "grounded"
            assert chat_payload["retrieval"]["packed_count"] >= 1
            assert chat_payload["retrieval"]["packed_count"] <= chat_payload["retrieval"]["returned"]
            assert chat_payload["retrieval"]["packed_token_estimate"] >= 1
            assert len(chat_payload["sources"]) >= 1
            assert len(chat_payload["sources"]) == chat_payload["retrieval"]["packed_count"]
            assert chat_payload["sources"][0]["document_id"] == document_id
            assert "refund" in chat_payload["sources"][0]["snippet"].lower()

            # P05-T02: Verify retrieval metadata completeness in chat response
            retrieval = chat_payload["retrieval"]
            assert retrieval["latency_ms"] >= 0
            assert retrieval["top_k"] >= 1
            assert retrieval["returned"] >= 1
            assert retrieval["request_id"] is not None
            # P06 (FINDING-R16): session_scope is required in retrieval metadata
            assert "session_scope" in retrieval, "session_scope must be present in retrieval metadata"
            assert retrieval["session_scope"] in ("session", "global", "all"), f"Unexpected session_scope: {retrieval['session_scope']}"
            assert retrieval["original_query"] is not None
            assert retrieval["rewritten_query"] is not None
            assert retrieval["query_expansions"] is not None

            # P05-T04: Verify source shape in chat response
            source_0 = chat_payload["sources"][0]
            assert "document_id" in source_0
            assert "chunk_id" in source_0
            assert "title" in source_0
            assert source_0["source_type"] == "knowledge"
            assert "source_uri" in source_0  # may be None for raw-text documents
            assert source_0["rank"] == 1

            history_response = client.get(
                api_path(f"/chat/sessions/{session_id}/messages"),
                headers=headers,
            )
            assert history_response.status_code == 200, history_response.text
            history = history_response.json()
            assert len(history) == 2
            assistant_message = history[-1]
            assert assistant_message["role"] == "assistant"
            assert assistant_message["request_id"] == chat_payload["request_id"]
            assert assistant_message["retrieval"]["used"] is True
            assert assistant_message["retrieval"]["strategy"] == "hybrid_rerank"
            assert assistant_message["retrieval"]["answer_policy"] == "grounded"
            assert "refund" in (assistant_message["retrieval"]["rewritten_query"] or "").lower()
            assert assistant_message["retrieval"]["packed_count"] >= 1
            assert len(assistant_message["sources"]) >= 1
            assert assistant_message["sources"][0]["title"] == "Product FAQ"

            # P05-T04: Verify citations are persisted and loadable in history
            assert len(assistant_message["sources"]) >= 1
            # The persisted sources should have chunk_id and document_id
            assert assistant_message["sources"][0].get("chunk_id") is not None
            assert assistant_message["sources"][0].get("document_id") == document_id
            # P05-T05: Retrieval metadata in history must include key fields
            history_retrieval = assistant_message.get("retrieval") or {}
            if history_retrieval.get("used"):
                assert history_retrieval.get("returned", 0) >= 1
                assert history_retrieval["strategy"] == "hybrid_rerank"
                assert history_retrieval["answer_policy"] == "grounded"

            weak_chat_response = client.post(
                api_path("/chat/"),
                json={
                    "session_id": session_id,
                    "message": "Bảo hành thiết bị trong bao lâu?",
                    "knowledge_document_id": document_id,
                },
                headers=headers,
            )
            assert weak_chat_response.status_code == 200, weak_chat_response.text
            weak_payload = weak_chat_response.json()
            assert weak_payload["retrieval"]["answer_policy"] == "insufficient_evidence"
            assert weak_payload["sources"] == []
            assert "chưa có đủ bằng chứng" in weak_payload["reply"].lower()

            section_ingest_response = client.post(
                api_path("/knowledge/documents/ingest"),
                json={
                    "title": "Synthetic Practice Section",
                    "source_type": "text",
                    "raw_text": SECTION_TEXT,
                    "metadata": {"category": "smoke", "language": "vi"},
                },
                headers=headers,
            )
            assert section_ingest_response.status_code == 201, section_ingest_response.text
            section_document_id = section_ingest_response.json()["document_id"]

            section_chat_response = client.post(
                api_path("/chat/"),
                json={
                    "session_id": session_id,
                    "message": "Bài thực hành số 4 có mấy bài, tóm tắt từng bài",
                    "knowledge_document_id": section_document_id,
                },
                headers=headers,
            )
            assert section_chat_response.status_code == 200, section_chat_response.text
            section_payload = section_chat_response.json()
            section_retrieval = section_payload["retrieval"]
            assert section_retrieval["used"] is True
            assert section_retrieval["rag_mode"] == "section_rag"
            assert section_retrieval["retrieval_scope"] == "document"
            assert section_retrieval["selected_document_id"] == section_document_id
            assert section_retrieval["section_key"] == "bai-thuc-hanh-so-4"
            assert section_retrieval["vector_store_attempted"] is False
            assert section_retrieval["returned"] >= 1
            assert section_payload["sources"]
            assert all(
                source["document_id"] == section_document_id
                and source.get("section_key") == "bai-thuc-hanh-so-4"
                for source in section_payload["sources"]
            )
            assert "3 bài" in section_payload["reply"]

    finally:
        settings.rate_limit_enabled = original_rate_limit_enabled
        chat_service.llm_provider.complete = original_complete
        app.dependency_overrides.clear()

    print("RAG_CHAT_SMOKE_OK")


if __name__ == "__main__":
    main()
