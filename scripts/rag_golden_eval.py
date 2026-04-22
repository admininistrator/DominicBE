from __future__ import annotations

import json
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


DATASET_PATH = PROJECT_ROOT / "scripts" / "data" / "rag_golden_set.json"


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


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def main() -> None:
    dataset = load_dataset()
    assert dataset, "Golden set must not be empty."

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
            failures: list[str] = []
            for index, case in enumerate(dataset, start=1):
                username = f"golden_user_{index}"
                register_response = client.post(
                    "/api/auth/register",
                    json={
                        "username": username,
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
                        "title": case["document_title"],
                        "source_type": "text",
                        "raw_text": case["document_text"],
                    },
                    headers=headers,
                )
                assert ingest_response.status_code == 201, ingest_response.text
                document_id = ingest_response.json()["document_id"]

                session_response = client.post(
                    "/api/chat/sessions",
                    json={"title": f"golden-{case['id']}"},
                    headers=headers,
                )
                assert session_response.status_code == 200, session_response.text
                session_id = session_response.json()["id"]

                payload = {
                    "session_id": session_id,
                    "message": case["message"],
                }
                if case.get("use_scope"):
                    payload["knowledge_document_id"] = document_id

                chat_response = client.post(
                    "/api/chat/",
                    json=payload,
                    headers=headers,
                )
                assert chat_response.status_code == 200, chat_response.text
                chat_payload = chat_response.json()
                retrieval = chat_payload.get("retrieval") or {}
                answer_policy = retrieval.get("answer_policy")
                sources_count = len(chat_payload.get("sources") or [])
                reply_text = (chat_payload.get("reply") or "").lower()

                if answer_policy != case["expected_answer_policy"]:
                    failures.append(
                        f"{case['id']}: expected policy={case['expected_answer_policy']} got {answer_policy}"
                    )
                if sources_count < int(case["expected_sources_min"]):
                    failures.append(
                        f"{case['id']}: expected at least {case['expected_sources_min']} sources got {sources_count}"
                    )
                if case["expected_reply_contains"].lower() not in reply_text:
                    failures.append(
                        f"{case['id']}: expected reply to contain '{case['expected_reply_contains']}' got '{chat_payload.get('reply')}'"
                    )

            if failures:
                raise AssertionError("\n".join(failures))

    finally:
        chat_service._get_client = original_get_client
        app.dependency_overrides.clear()

    print(f"RAG_GOLDEN_EVAL_OK cases={len(dataset)}")


if __name__ == "__main__":
    main()


