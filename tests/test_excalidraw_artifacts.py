from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.services import chat_service
from app.services.artifacts.diagram_intent import is_diagram_intent
from app.services.artifacts.excalidraw_schema import (
    ExcalidrawValidationError,
    normalize_excalidraw_elements,
)


def test_diagram_intent_detects_vietnamese_and_english_requests():
    assert is_diagram_intent("vẽ sơ đồ kiến trúc frontend -> backend -> qdrant")
    assert is_diagram_intent("ve so do luong dang nhap")
    assert is_diagram_intent("draw an architecture diagram for the RAG pipeline")
    assert is_diagram_intent("please create a sequence diagram")
    assert is_diagram_intent("vẽ UML cho chức năng thanh toán")


def test_diagram_intent_does_not_route_normal_chat():
    assert not is_diagram_intent("tell me about system design tradeoffs")
    assert not is_diagram_intent("draw conclusions from this paragraph")
    assert not is_diagram_intent("what is the refund policy?")


def test_excalidraw_schema_accepts_and_normalizes_valid_elements():
    elements = normalize_excalidraw_elements([
        {"type": "cameraUpdate", "x": 0, "y": 0, "width": 800, "height": 600},
        {"type": "rectangle", "x": 10, "y": 20, "width": 120, "height": 60},
        {"type": "text", "x": 20, "y": 35, "width": 100, "height": 24, "text": "API"},
        {"type": "arrow", "x": 130, "y": 50, "width": 120, "height": 0},
    ])

    assert elements[0]["type"] == "cameraUpdate"
    assert elements[1]["id"].startswith("el_")
    assert elements[2]["originalText"] == "API"
    assert elements[3]["points"] == [[0.0, 0.0], [120.0, 0.0]]


def test_excalidraw_schema_rejects_malformed_oversized_and_unsupported_payloads():
    with pytest.raises(ExcalidrawValidationError):
        normalize_excalidraw_elements({"type": "rectangle"})
    with pytest.raises(ExcalidrawValidationError):
        normalize_excalidraw_elements([{"type": "iframe", "x": 0, "y": 0, "width": 1, "height": 1}])
    with pytest.raises(ExcalidrawValidationError):
        normalize_excalidraw_elements(
            [{"type": "text", "x": 0, "y": 0, "width": 1, "height": 1, "text": "x" * 128}],
            max_payload_bytes=64,
        )


def _make_prepared():
    prepared = MagicMock()
    prepared.sources = []
    prepared.retrieval_result = None
    prepared.web_search_result = {"used": False, "results": []}
    prepared.knowledge_document_id = None
    prepared.knowledge_base_active = False
    prepared.request_id = "req-native"
    prepared.model = None
    prepared.reasoning_effort = None
    prepared.username = "testuser"
    prepared.session_id = 1
    prepared.session = MagicMock()
    prepared.user_message = "draw an architecture diagram"
    prepared.existing_message_count = 0
    prepared.user_msg_id = 999
    prepared.request_kwargs = {"messages": [{"role": "user", "content": prepared.user_message}]}
    return prepared


@contextmanager
def _mock_finalize_dependencies():
    patches = (
        patch("app.services.chat_service.crud_chat.update_message_tokens_and_status"),
        patch("app.services.chat_service.crud_chat.create_message"),
        patch("app.services.chat_service.crud_chat.touch_chat_session"),
        patch("app.services.chat_service.crud_chat.increment_user_tokens"),
        patch("app.services.chat_service.crud_knowledge.update_retrieval_event_metadata"),
        patch("app.services.chat_service.crud_knowledge.replace_answer_citations"),
        patch("app.services.chat_service._maybe_autotitle_session"),
        patch("app.services.chat_service._linkify_web_sources_in_reply", side_effect=lambda content, sources: content),
        patch("app.services.chat_service._build_retrieval_payload", return_value=None),
        patch("app.services.chat_service._build_web_search_payload", return_value=None),
    )
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        yield


def test_native_excalidraw_stream_events_are_ordered_and_final_artifact_is_persistable():
    prepared = _make_prepared()
    raw = json.dumps([
        {"type": "cameraUpdate", "x": 0, "y": 0, "width": 800, "height": 500},
        {"type": "rectangle", "id": "frontend", "x": 20, "y": 40, "width": 160, "height": 72},
        {"type": "text", "id": "frontend-label", "x": 35, "y": 62, "width": 130, "height": 24, "text": "Frontend"},
    ])

    with (
        _mock_finalize_dependencies(),
        patch("app.services.chat_service._prepare_chat_turn", return_value=prepared),
        patch(
            "app.services.chat_service.llm_provider.stream_complete",
            return_value=iter([
                {"type": "delta", "text": raw[:60]},
                {"type": "delta", "text": raw[60:150]},
                {"type": "delta", "text": raw[150:]},
                {"type": "complete", "text": raw, "input_tokens": 10, "output_tokens": 20},
            ]),
        ),
    ):
        events = list(chat_service.handle_chat_stream(MagicMock(), "testuser", 1, prepared.user_message))

    assert [event["event"] for event in events if event["event"].startswith("artifact_")] == [
        "artifact_start",
        "artifact_delta",
        "artifact_delta",
        "artifact_done",
    ]
    assert all(event["event"] != "delta" for event in events)
    done = next(event for event in events if event["event"] == "artifact_done")
    assert done["data"]["kind"] == "excalidraw"
    assert done["data"]["elements"][0]["type"] == "cameraUpdate"
    final = events[-1]
    assert final["event"] == "final"
    assert final["data"]["artifacts"][0]["id"] == "excalidraw_req-native"


def test_native_excalidraw_invalid_final_json_emits_artifact_error():
    prepared = _make_prepared()
    bad = "not-json"

    with (
        _mock_finalize_dependencies(),
        patch("app.services.chat_service._prepare_chat_turn", return_value=prepared),
        patch(
            "app.services.chat_service.llm_provider.stream_complete",
            return_value=iter([
                {"type": "delta", "text": bad},
                {"type": "complete", "text": bad, "input_tokens": 10, "output_tokens": 20},
            ]),
        ),
    ):
        events = list(chat_service.handle_chat_stream(MagicMock(), "testuser", 1, prepared.user_message))

    assert any(event["event"] == "artifact_error" for event in events)
    assert events[-1]["event"] == "final"
    assert "artifacts" not in events[-1]["data"]

