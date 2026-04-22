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
from app.core.database import Base
from app.main import app
from app.services import chat_service


LONG_TEXT = (
    "Dominic Product FAQ. Refund policy requests are reviewed within 5 business days. "
    "Customers can submit refund evidence through the support portal. " * 30
).strip()


class _FakeTokenInfo:
    def __init__(self, input_tokens: int):
        self.input_tokens = input_tokens


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, *, input_tokens: int = 100, output_tokens: int = 40):
        self.content = [_FakeContentBlock(text)]
        self.usage = _FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeMessagesAPI:
    def count_tokens(self, **kwargs):
        total_chars = len(kwargs.get("system") or "")
        total_chars += sum(len((msg.get("content") or "")) for msg in kwargs.get("messages", []))
        return _FakeTokenInfo(max(1, total_chars // 4))

    def create(self, **kwargs):
        return _FakeResponse(
            "Theo Product FAQ, yêu cầu hoàn tiền được xem xét trong vòng 5 ngày làm việc. [Source 1]"
        )


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessagesAPI()


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

    original_get_client = chat_service._get_client
    app.dependency_overrides[get_db] = override_get_db
    chat_service._get_client = lambda: _FakeAnthropicClient()

    try:
        with TestClient(app) as client:
            register_response = client.post(
                "/api/auth/register",
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
                "/api/knowledge/documents/ingest",
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
                "/api/knowledge/search",
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
                "/api/chat/sessions",
                json={"title": "Eval chat"},
                headers=headers,
            )
            assert session_response.status_code == 200, session_response.text
            session_id = session_response.json()["id"]

            chat_response = client.post(
                "/api/chat/",
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
                "/api/chat/",
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
                "/api/knowledge/admin/analytics",
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
                "/api/knowledge/admin/analytics",
                params={"username": "test_user", "recent_limit": 5},
                headers=headers,
            )
            assert filtered_analytics_response.status_code == 200, filtered_analytics_response.text
            filtered_payload = filtered_analytics_response.json()
            assert filtered_payload["username_filter"] == "test_user"
            assert filtered_payload["summary"]["total_events"] >= 2

    finally:
        chat_service._get_client = original_get_client
        app.dependency_overrides.clear()

    print("RAG_EVAL_SMOKE_OK")


if __name__ == "__main__":
    main()

