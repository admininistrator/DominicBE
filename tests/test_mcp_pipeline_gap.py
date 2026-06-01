"""Positive integration tests for MCP artifact flow through chat service."""
from __future__ import annotations

import inspect
import json
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock

from app.services import chat_service
from app.services.chat_service import _finalize_chat_turn
from app.services.mcp.artifact import Artifact, McpToolResult


def _make_mock_prepared():
    prepared = MagicMock()
    prepared.sources = []
    prepared.retrieval_result = None
    prepared.web_search_result = {"used": False, "results": []}
    prepared.knowledge_document_id = None
    prepared.knowledge_base_active = False
    prepared.request_id = "req-123"
    prepared.model = None
    prepared.reasoning_effort = None
    prepared.username = "testuser"
    prepared.session_id = 1
    prepared.session = MagicMock()
    prepared.user_message = "test"
    prepared.existing_message_count = 0
    prepared.user_msg_id = 999
    prepared.request_kwargs = {"messages": [{"role": "user", "content": prepared.user_message}]}
    return prepared


class _FakeExcalidrawConfig:
    def get_server(self, server_id):
        return MagicMock(id=server_id, enabled=True) if server_id == "excalidraw" else None


class _FakeExcalidrawManager:
    enabled = True
    config = _FakeExcalidrawConfig()

    def __init__(self):
        self.global_config = MagicMock(max_artifact_content_bytes=512000)
        self.calls = []

    def is_tool_allowed(self, server_id, tool_name):
        return server_id == "excalidraw" and tool_name in {"export_to_excalidraw", "create_view"}

    async def invoke_tool(self, server_id, tool_name, arguments, **kwargs):
        self.calls.append((server_id, tool_name, arguments, kwargs))
        if tool_name == "create_view":
            return McpToolResult(
                server_id=server_id,
                tool_name=tool_name,
                status="success",
                duration_ms=12,
                raw_content={
                    "content": [{"type": "text", "text": 'Diagram displayed! Checkpoint id: "abc".'}],
                    "structuredContent": {"checkpointId": "abc"},
                },
            )
        return McpToolResult(
            server_id=server_id,
            tool_name=tool_name,
            status="success",
            duration_ms=12,
            raw_content={"content": [{"type": "text", "text": "https://excalidraw.com/#json=mock"}]},
        )


class _FakeCreateViewOnlyManager(_FakeExcalidrawManager):
    def is_tool_allowed(self, server_id, tool_name):
        return server_id == "excalidraw" and tool_name == "create_view"

    async def invoke_tool(self, server_id, tool_name, arguments, **kwargs):
        self.calls.append((server_id, tool_name, arguments, kwargs))
        return McpToolResult(
            server_id=server_id,
            tool_name=tool_name,
            status="success",
            duration_ms=12,
            raw_content={"content": [{"type": "text", "text": 'Diagram displayed! Checkpoint id: "abc".'}]},
        )


class _FakeConnectionErrorManager(_FakeExcalidrawManager):
    async def invoke_tool(self, server_id, tool_name, arguments, **kwargs):
        self.calls.append((server_id, tool_name, arguments, kwargs))
        return McpToolResult(
            server_id=server_id,
            tool_name=tool_name,
            status="connection_error",
            duration_ms=3,
            error="MCP Python SDK is not installed",
        )


@contextmanager
def _mock_finalize_dependencies(patch):
    patches = (
        patch("app.services.chat_service.crud_chat.update_message_tokens_and_status"),
        patch("app.services.chat_service.crud_chat.create_message"),
        patch("app.services.chat_service.crud_chat.touch_chat_session"),
        patch("app.services.chat_service.crud_chat.increment_user_tokens"),
        patch("app.services.chat_service.crud_knowledge.update_retrieval_event_metadata"),
        patch("app.services.chat_service.crud_knowledge.replace_answer_citations"),
        patch("app.services.chat_service._maybe_autotitle_session"),
        patch(
            "app.services.chat_service._apply_answer_guardrails",
            side_effect=lambda content, retrieval_result, sources, *args, **kwargs: (content, sources, "grounded"),
        ),
        patch("app.services.chat_service._linkify_web_sources_in_reply", side_effect=lambda content, sources: content),
        patch("app.services.chat_service._build_retrieval_payload", return_value=None),
        patch("app.services.chat_service._build_web_search_payload", return_value=None),
    )
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        yield


def test_finalize_chat_turn_includes_mcp_artifacts_when_provided():
    from unittest.mock import patch

    prepared = _make_mock_prepared()
    safe_art = Artifact(
        id="art_1",
        type="excalidraw",
        title="Diagram",
        url="https://excalidraw.com/#json=abc",
        tool_server_id="excalidraw",
        tool_name="export_to_excalidraw",
        safe=True,
    )
    tool_result = McpToolResult(
        server_id="excalidraw",
        tool_name="export_to_excalidraw",
        status="success",
        duration_ms=1200,
        artifact_ids=["art_1"],
    )

    with _mock_finalize_dependencies(patch):
        result = _finalize_chat_turn(
            MagicMock(),
            prepared,
            ai_content="Here is your diagram",
            input_tokens=10,
            output_tokens=20,
            mcp_artifacts=[safe_art],
            mcp_tool_results=[tool_result],
        )

    assert result["artifacts"][0]["id"] == "art_1"
    assert result["artifacts"][0]["type"] == "excalidraw"
    assert result["tool_results"][0]["status"] == "success"
    assert result["reply"] == "Here is your diagram"
    assert result["request_id"] == "req-123"
    assert "usage" in result


def test_finalize_chat_turn_omits_mcp_fields_when_not_provided():
    from unittest.mock import patch

    prepared = _make_mock_prepared()
    with _mock_finalize_dependencies(patch):
        result = _finalize_chat_turn(
            MagicMock(),
            prepared,
            ai_content="Hello",
            input_tokens=10,
            output_tokens=20,
        )

    assert "artifacts" not in result
    assert "tool_results" not in result
    assert result["reply"] == "Hello"


def test_handle_chat_receives_mcp_client_manager_parameter():
    sig = inspect.signature(chat_service.handle_chat)
    assert "mcp_client_manager" in sig.parameters


def test_handle_chat_stream_receives_mcp_client_manager_parameter():
    sig = inspect.signature(chat_service.handle_chat_stream)
    assert "mcp_client_manager" in sig.parameters


def test_handle_chat_stream_invokes_excalidraw_and_final_event_has_artifacts():
    from unittest.mock import patch

    prepared = _make_mock_prepared()
    prepared.user_message = "Please draw an Excalidraw architecture diagram"
    manager = _FakeExcalidrawManager()
    scene = {"type": "excalidraw", "version": 2, "elements": []}

    with (
        _mock_finalize_dependencies(patch),
        patch("app.services.chat_service._prepare_chat_turn", return_value=prepared),
        patch(
            "app.services.chat_service.llm_provider.stream_complete",
            return_value=iter([
                {"type": "delta", "text": json.dumps(scene)},
                {"type": "complete", "text": json.dumps(scene), "input_tokens": 10, "output_tokens": 20},
            ]),
        ),
    ):
        events = list(chat_service.handle_chat_stream(
            MagicMock(),
            "testuser",
            1,
            prepared.user_message,
            mcp_client_manager=manager,
        ))

    assert manager.calls
    server_id, tool_name, arguments, kwargs = manager.calls[0]
    assert server_id == "excalidraw"
    assert tool_name == "create_view"
    assert json.loads(arguments["elements"])
    assert kwargs["turn_id"] == "req-123"

    final_event = events[-1]
    assert final_event["event"] == "final"
    assert final_event["data"]["success"] is True
    assert final_event["data"]["reply"] == "I created the Excalidraw diagram. Open the artifact below to view or edit it."
    assert final_event["data"]["artifacts"][0]["type"] == "excalidraw"
    assert final_event["data"]["artifacts"][0]["url"] is None
    assert json.loads(final_event["data"]["artifacts"][0]["content"])["type"] == "excalidraw"
    assert final_event["data"]["artifacts"][0]["metadata"]["checkpoint_id"] == "abc"
    assert final_event["data"]["tool_results"][0]["tool_name"] == "create_view"
    assert final_event["data"]["tool_results"][0]["artifact_ids"] == [
        final_event["data"]["artifacts"][0]["id"]
    ]


def test_handle_chat_without_mcp_omits_artifacts_and_tool_results():
    from unittest.mock import patch

    prepared = _make_mock_prepared()
    prepared.user_message = "Hello"

    with (
        _mock_finalize_dependencies(patch),
        patch("app.services.chat_service._prepare_chat_turn", return_value=prepared),
        patch(
            "app.services.chat_service.llm_provider.complete",
            return_value={"text": "Hello there", "input_tokens": 3, "output_tokens": 4},
        ),
    ):
        result = chat_service.handle_chat(
            MagicMock(),
            "testuser",
            1,
            prepared.user_message,
            mcp_client_manager=None,
        )

    assert result["reply"] == "Hello there"
    assert "artifacts" not in result
    assert "tool_results" not in result


def test_handle_chat_stream_create_view_checkpoint_result_gets_inline_artifact_fallback():
    from unittest.mock import patch

    prepared = _make_mock_prepared()
    prepared.user_message = "Please draw an Excalidraw flowchart"
    manager = _FakeCreateViewOnlyManager()

    with (
        _mock_finalize_dependencies(patch),
        patch("app.services.chat_service._prepare_chat_turn", return_value=prepared),
        patch(
            "app.services.chat_service.llm_provider.stream_complete",
            return_value=iter([
                {"type": "complete", "text": "Flowchart summary", "input_tokens": 10, "output_tokens": 20},
            ]),
        ),
    ):
        events = list(chat_service.handle_chat_stream(
            MagicMock(),
            "testuser",
            1,
            prepared.user_message,
            mcp_client_manager=manager,
        ))

    assert manager.calls[0][1] == "create_view"
    final_data = events[-1]["data"]
    assert final_data["artifacts"][0]["type"] == "excalidraw"
    assert final_data["artifacts"][0]["url"] is None
    assert json.loads(final_data["artifacts"][0]["content"])["type"] == "excalidraw"
    assert final_data["tool_results"][0]["artifact_ids"] == [final_data["artifacts"][0]["id"]]


def test_handle_chat_stream_connection_error_still_returns_inline_excalidraw_artifact():
    from unittest.mock import patch

    prepared = _make_mock_prepared()
    prepared.user_message = "Ve so do use case he thong ban hang bang Excalidraw"
    manager = _FakeConnectionErrorManager()

    with (
        _mock_finalize_dependencies(patch),
        patch("app.services.chat_service._prepare_chat_turn", return_value=prepared),
        patch(
            "app.services.chat_service.llm_provider.stream_complete",
            return_value=iter([
                {"type": "complete", "text": "Use case summary", "input_tokens": 10, "output_tokens": 20},
            ]),
        ),
    ):
        events = list(chat_service.handle_chat_stream(
            MagicMock(),
            "testuser",
            1,
            prepared.user_message,
            mcp_client_manager=manager,
        ))

    assert [call[1] for call in manager.calls] == ["create_view"]
    final_data = events[-1]["data"]
    assert final_data["artifacts"][0]["type"] == "excalidraw"
    assert final_data["artifacts"][0]["url"] is None
    scene = json.loads(final_data["artifacts"][0]["content"])
    assert scene["type"] == "excalidraw"
    assert len(scene["elements"]) > 10
    assert final_data["tool_results"][-1]["status"] == "connection_error"
    assert final_data["tool_results"][-1]["artifact_ids"] == [final_data["artifacts"][0]["id"]]
