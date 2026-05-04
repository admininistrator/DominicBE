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
    "Dominic Product FAQ. Refund policy requests are reviewed within 5 business days. "
    "Customers can submit refund evidence through the support portal. " * 30
).strip()

def _fake_complete(*, messages, system=None, max_tokens=1024, **kwargs):
    assert messages
    assert system
    assert max_tokens >= 1
    return {
        "text": "Theo Product FAQ, yêu cầu hoàn tiền được xem xét trong vòng 5 ngày làm việc. [Source 1]",
        "input_tokens": 100,
        "output_tokens": 40,
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
                    "username": "test_user",
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

            search_response = client.post(
                api_path("/knowledge/search"),
                json={
                    "query": "Chính sách hoàn tiền xử lý bao lâu?",
                    "top_k": 3,
                    "document_id": document_id,
                },
                headers=headers,
            )
            assert search_response.status_code == 200, search_response.text
            search_payload = search_response.json()
            assert search_payload["strategy"] == "hybrid_rerank"
            assert search_payload["returned"] >= 1

            session_response = client.post(
                api_path("/chat/sessions"),
                json={"title": "Eval chat"},
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
            assert chat_payload["sources"]
            assert chat_payload["retrieval"]["answer_policy"] == "grounded"

            insufficient_response = client.post(
                api_path("/chat/"),
                json={
                    "session_id": session_id,
                    "message": "Bảo hành thiết bị trong bao lâu?",
                    "knowledge_document_id": document_id,
                },
                headers=headers,
            )
            assert insufficient_response.status_code == 200, insufficient_response.text
            insufficient_payload = insufficient_response.json()
            assert insufficient_payload["retrieval"]["answer_policy"] == "insufficient_evidence"
            assert insufficient_payload["sources"] == []

            analytics_response = client.get(
                api_path("/knowledge/admin/analytics"),
                params={"recent_limit": 10},
                headers=headers,
            )
            assert analytics_response.status_code == 200, analytics_response.text
            analytics_payload = analytics_response.json()
            assert analytics_payload["summary"]["total_events"] >= 2
            assert analytics_payload["summary"]["hit_rate"] > 0
            assert analytics_payload["summary"]["avg_results_returned"] > 0
            assert analytics_payload["summary"]["avg_citations_per_answer"] > 0
            assert analytics_payload["summary"]["indexed_documents"] >= 1
            assert analytics_payload["summary"]["total_chunks"] >= 1
            assert analytics_payload["summary"]["answer_policy_counts"]["grounded"] >= 1
            assert analytics_payload["summary"]["answer_policy_counts"]["insufficient_evidence"] >= 1
            assert analytics_payload["summary"]["evidence_strength_counts"]["grounded"] >= 1
            assert analytics_payload["summary"]["scoped_events"] >= 2
            assert analytics_payload["summary"]["scoped_rate"] > 0
            assert analytics_payload["summary"]["citationless_grounded_count"] == 0
            assert len(analytics_payload["recent_events"]) >= 2
            assert any(event["strategy"] == "hybrid_rerank" for event in analytics_payload["recent_events"])
            assert any(event["citations_count"] >= 1 for event in analytics_payload["recent_events"])
            assert any(event["answer_policy"] == "grounded" for event in analytics_payload["recent_events"])
            assert any(event["answer_policy"] == "insufficient_evidence" for event in analytics_payload["recent_events"])
            assert any(event["scoped"] is True for event in analytics_payload["recent_events"])

            filtered_analytics_response = client.get(
                api_path("/knowledge/admin/analytics"),
                params={"username": "test_user", "recent_limit": 5},
                headers=headers,
            )
            assert filtered_analytics_response.status_code == 200, filtered_analytics_response.text
            filtered_payload = filtered_analytics_response.json()
            assert filtered_payload["username_filter"] == "test_user"
            assert filtered_payload["summary"]["total_events"] >= 2

    finally:
        settings.rate_limit_enabled = original_rate_limit_enabled
        chat_service.llm_provider.complete = original_complete
        app.dependency_overrides.clear()

    print("RAG_EVAL_SMOKE_OK")


if __name__ == "__main__":
    main()

