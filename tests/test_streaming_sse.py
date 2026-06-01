from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

os.environ["DEBUG"] = "false"

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.endpoints import chat
from app.main import app
import app.main as main_module


@pytest.fixture()
def app_instance(monkeypatch):
    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", noop_lifespan)
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app.dependency_overrides.clear()
    yield app
    app.dependency_overrides.clear()


def _override_current_user(username: str = "tester", role: str = "user"):
    return lambda: SimpleNamespace(username=username, role=role)


def _payload() -> dict:
    return {
        "username": "tester",
        "session_id": 1,
        "message": "Stream a deterministic response.",
        "knowledge_document_id": None,
        "use_web_search": False,
        "images": [],
        "image_media_types": [],
    }


def _parse_sse_events(stream_text: str) -> list[dict]:
    events: list[dict] = []
    for raw_event in stream_text.strip().split("\n\n"):
        if not raw_event.strip():
            continue
        event_name = None
        data_lines: list[str] = []
        for line in raw_event.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))
        assert event_name is not None, raw_event
        assert data_lines, raw_event
        events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
    return events


def _request_ids(events: list[dict]) -> list[str]:
    ids: list[str] = []
    for event in events:
        request_id = event["data"].get("request_id")
        if request_id is not None:
            ids.append(request_id)
    return ids


def test_streaming_sse_contract_headers_ordering_and_request_id(app_instance, monkeypatch):
    request_id = "req-stream-123"

    def fake_handle_chat_stream(*args, **kwargs):
        yield {
            "event": "start",
            "data": {
                "request_id": request_id,
                "rag_mode": "direct_chat",
                "retrieval_scope": "none",
                "sources": [],
                "has_web_search": False,
            },
        }
        yield {"event": "delta", "data": {"text": "Hello", "request_id": request_id}}
        yield {"event": "delta", "data": {"text": " world", "request_id": request_id}}
        yield {
            "event": "final",
            "data": {
                "success": True,
                "reply": "Hello world",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                "request_id": request_id,
                "sources": [],
                "retrieval": None,
            },
        }

    monkeypatch.setattr(chat, "handle_chat_stream", fake_handle_chat_stream)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        with client.stream("POST", "/api/v1/chat/stream", json=_payload()) as response:
            stream_text = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"

    events = _parse_sse_events(stream_text)
    assert [event["event"] for event in events] == ["start", "delta", "delta", "final"]
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "final"
    assert all(event["event"] != "delta" for event in events[:1])
    final_index = next(index for index, event in enumerate(events) if event["event"] == "final")
    assert all(event["event"] != "delta" for event in events[final_index + 1 :])
    assert events[1]["data"]["text"] == "Hello"
    assert events[2]["data"]["text"] == " world"
    assert events[-1]["data"]["success"] is True
    assert events[-1]["data"]["reply"] == "Hello world"
    assert _request_ids(events) == [request_id, request_id, request_id, request_id]
    assert events[0]["data"]["request_id"] == request_id
    assert events[0]["data"]["rag_mode"] == "direct_chat"
    assert events[0]["data"]["retrieval_scope"] == "none"
    assert events[0]["data"]["sources"] == []
    assert events[0]["data"]["has_web_search"] is False


def test_streaming_sse_error_event_is_sanitized_and_terminal(app_instance, monkeypatch):
    secret = "super-secret-provider-token"
    request_id = "req-error-456"

    def fake_handle_chat_stream(*args, **kwargs):
        yield {
            "event": "start",
            "data": {
                "request_id": request_id,
                "rag_mode": "session_rag",
                "retrieval_scope": "session",
                "sources": [
                    {
                        "document_id": 7,
                        "chunk_id": 11,
                        "title": "Safe document title",
                        "source_type": "knowledge",
                        "rank": 1,
                    }
                ],
                "has_web_search": True,
            },
        }
        raise RuntimeError(secret)
        yield

    monkeypatch.setattr(chat, "handle_chat_stream", fake_handle_chat_stream)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        with client.stream("POST", "/api/v1/chat/stream", json=_payload()) as response:
            stream_text = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert secret not in stream_text

    events = _parse_sse_events(stream_text)
    assert [event["event"] for event in events] == ["start", "error"]
    assert events[0]["data"]["request_id"] == request_id
    assert events[0]["data"]["rag_mode"] == "session_rag"
    assert events[0]["data"]["retrieval_scope"] == "session"
    assert events[0]["data"]["sources"] == [
        {
            "document_id": 7,
            "chunk_id": 11,
            "title": "Safe document title",
            "source_type": "knowledge",
            "rank": 1,
        }
    ]
    assert events[0]["data"]["has_web_search"] is True
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["status_code"] == 500
    assert events[-1]["data"]["detail"] == "An internal server error occurred."
    assert all(event["event"] != "delta" for event in events)
