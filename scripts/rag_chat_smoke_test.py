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
    "Dominic Product FAQ. Refund requests are reviewed within 5 business days. "
    "Customers can submit refund evidence through the support portal. " * 40
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
    def __init__(self, text: str, *, input_tokens: int = 120, output_tokens: int = 48):
        self.content = [_FakeContentBlock(text)]
        self.usage = _FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeMessagesAPI:
    def count_tokens(self, **kwargs):
        total_chars = len(kwargs.get("system") or "")
        total_chars += sum(len((msg.get("content") or "")) for msg in kwargs.get("messages", []))
        return _FakeTokenInfo(max(1, total_chars // 4))

    def create(self, **kwargs):
        system_prompt = kwargs.get("system") or ""
        assert "Knowledge-base evidence for this turn" in system_prompt
        assert "Product FAQ" in system_prompt
        assert "refund requests are reviewed within 5 business days" in system_prompt.lower()
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
                    "username": "rag_user",
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

            session_response = client.post(
                "/api/chat/sessions",
                json={"title": "Grounded chat"},
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

            history_response = client.get(
                f"/api/chat/sessions/{session_id}/messages",
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

            weak_chat_response = client.post(
                "/api/chat/",
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

    finally:
        chat_service._get_client = original_get_client
        app.dependency_overrides.clear()

    print("RAG_CHAT_SMOKE_OK")


if __name__ == "__main__":
    main()

