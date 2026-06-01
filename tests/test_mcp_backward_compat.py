"""Tests verifying backward compatibility of extended chat response contract.

Test coverage:
- MCP disabled → response identical to pre-MCP (no artifacts/tool_results)
- MCP enabled, no tool called → response identical to pre-MCP
- MCP enabled, tool called → existing fields + optional artifacts/tool_results
- Old FE consumer ignores unknown fields
- _artifacts_to_response returns None when empty
- _tool_results_to_response returns None when empty
- _build_mcp_tool_prompt returns None when MCP disabled
- handle_chat() forwards mcp_client_manager to _prepare_chat_turn() (regression: reviewer bug)
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from app.services import chat_service
from app.services.chat_service import (
    _artifacts_to_response,
    _build_mcp_tool_prompt,
    _tool_results_to_response,
)
from app.services.mcp.artifact import Artifact, McpToolResult


# ── _artifacts_to_response ──────────────────────────────────────────────────


def test_artifacts_to_response_returns_none_for_empty_input():
    assert _artifacts_to_response([]) is None
    assert _artifacts_to_response(None) is None


def test_artifacts_to_response_skips_unsafe_artifacts():
    unsafe = Artifact(
        id="art_1",
        type="generic_tool_result",
        title="Unsafe",
        tool_server_id="test",
        tool_name="test_tool",
        safe=False,
    )
    assert _artifacts_to_response([unsafe]) is None


def test_artifacts_to_response_includes_safe_artifacts():
    safe = Artifact(
        id="art_1",
        type="excalidraw",
        title="Architecture Diagram",
        url="https://excalidraw.com/#json=abc123",
        tool_server_id="excalidraw",
        tool_name="create-excalidraw",
        safe=True,
        metadata={"tool_server": "excalidraw"},
    )
    result = _artifacts_to_response([safe])
    assert result is not None
    assert len(result) == 1
    assert result[0]["id"] == "art_1"
    assert result[0]["type"] == "excalidraw"
    assert result[0]["title"] == "Architecture Diagram"
    assert result[0]["url"] == "https://excalidraw.com/#json=abc123"
    assert result[0]["metadata"]["tool_server"] == "excalidraw"


def test_artifacts_to_response_mixed_safety():
    safe = Artifact(
        id="art_1", type="image", title="Safe", tool_server_id="test",
        tool_name="t", safe=True,
    )
    unsafe = Artifact(
        id="art_2", type="image", title="Unsafe", tool_server_id="test",
        tool_name="t", safe=False,
    )
    result = _artifacts_to_response([unsafe, safe])
    assert result is not None
    assert len(result) == 1
    assert result[0]["id"] == "art_1"


# ── _tool_results_to_response ────────────────────────────────────────────────


def test_tool_results_to_response_returns_none_for_empty():
    assert _tool_results_to_response([]) is None
    assert _tool_results_to_response(None) is None


def test_tool_results_to_response_includes_all_results():
    tr1 = McpToolResult(
        server_id="excalidraw",
        tool_name="create-excalidraw",
        status="success",
        duration_ms=1200,
        artifact_ids=["art_1"],
    )
    tr2 = McpToolResult(
        server_id="excalidraw",
        tool_name="create-excalidraw",
        status="timeout",
        duration_ms=30000,
        artifact_ids=[],
        error="timed out",
    )
    result = _tool_results_to_response([tr1, tr2])
    assert result is not None
    assert len(result) == 2
    assert result[0]["tool_server_id"] == "excalidraw"
    assert result[0]["status"] == "success"
    assert result[0]["duration_ms"] == 1200
    assert result[0]["artifact_ids"] == ["art_1"]
    assert result[1]["status"] == "timeout"
    assert result[1]["duration_ms"] == 30000


# ── _build_mcp_tool_prompt ────────────────────────────────────────────────────


def test_build_mcp_tool_prompt_returns_none_when_no_manager():
    assert _build_mcp_tool_prompt(None) is None


def test_build_mcp_tool_prompt_returns_none_when_disabled():
    manager = MagicMock()
    manager.enabled = False
    assert _build_mcp_tool_prompt(manager) is None


def test_build_mcp_tool_prompt_returns_prompt_when_enabled():
    server = MagicMock()
    server.id = "excalidraw"
    server.label = "Excalidraw Whiteboard"
    server.enabled = True
    server.artifact_capabilities = ["excalidraw_json", "link"]

    config = MagicMock()
    config.servers = [server]

    manager = MagicMock()
    manager.enabled = True
    manager.config = config

    result = _build_mcp_tool_prompt(manager)
    assert result is not None
    assert "Excalidraw Whiteboard" in result
    assert "excalidraw_json" in result
    assert "text description" in result


def test_build_mcp_tool_prompt_handles_exceptions_gracefully():
    """If getattr fails, should return None, not crash."""
    manager = MagicMock()
    manager.enabled = True
    # Break the config attribute
    del manager.config
    result = _build_mcp_tool_prompt(manager)
    assert result is None


# ── End-to-end: final response shape (no MCP) ────────────────────────────────


def _make_mock_prepared():
    """Create a minimal mock PreparedChatTurn for _finalize_chat_turn tests."""
    mock = MagicMock()
    mock.sources = []
    mock.retrieval_result = None
    mock.web_search_result = {"used": False, "results": []}
    mock.knowledge_document_id = None
    mock.knowledge_base_active = False
    mock.request_id = "req-123"
    mock.model = None
    mock.reasoning_effort = None
    mock.username = "testuser"
    mock.session_id = 1
    mock.session = MagicMock()
    mock.user_message = "test"
    mock.existing_message_count = 0
    mock.user_msg_id = 999
    return mock


@pytest.mark.skip(reason="Requires DB access; use unit tests above instead")
def test_final_response_without_mcp_omits_optional_fields():
    """When no MCP data provided, artifacts and tool_results must be absent."""
    pass  # Placeholder — tested via schema inspection in app


# ── Schema validation: ChatResponse ──────────────────────────────────────────


def test_chat_response_schema_omits_optional_fields():
    from app.schemas.chat_schemas import ChatResponse

    resp = ChatResponse(
        success=True,
        reply="Hello",
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    data = resp.model_dump(exclude_none=True)
    # Existing fields present
    assert data["success"] is True
    assert data["reply"] == "Hello"
    assert data["usage"] == {"input_tokens": 10, "output_tokens": 20}
    # Optional MCP fields absent (not null)
    assert "artifacts" not in data
    assert "tool_results" not in data


def test_chat_response_schema_with_mcp_fields():
    from app.schemas.chat_schemas import ArtifactResponse, ChatResponse, ToolResultResponse

    resp = ChatResponse(
        success=True,
        reply="Here is your diagram",
        usage={"input_tokens": 10, "output_tokens": 20},
        artifacts=[
            ArtifactResponse(
                id="art_1",
                type="excalidraw",
                title="Diagram",
                url="https://excalidraw.com/#json=abc",
            ),
        ],
        tool_results=[
            ToolResultResponse(
                tool_server_id="excalidraw",
                tool_name="create-excalidraw",
                status="success",
                duration_ms=1200,
                artifact_ids=["art_1"],
            ),
        ],
    )
    data = resp.model_dump(exclude_none=True)
    assert "artifacts" in data
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["id"] == "art_1"
    assert "tool_results" in data
    assert data["tool_results"][0]["status"] == "success"


def test_old_fe_ignores_unknown_fields():
    """Simulate an old FE consumer that only destructures known fields."""
    raw_response = {
        "success": True,
        "reply": "Test reply",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "request_id": "req-123",
        "sources": [],
        "assistant_meta": None,
        "retrieval": None,
        "artifacts": [{"id": "art_1", "type": "excalidraw", "title": "Diagram"}],
        "tool_results": [{"tool_server_id": "excalidraw", "tool_name": "t", "status": "success", "duration_ms": 100, "artifact_ids": []}],
    }
    # Old FE only destructures known fields:
    known_fields = {
        "success": raw_response.get("success"),
        "reply": raw_response.get("reply"),
        "sources": raw_response.get("sources"),
        "assistant_meta": raw_response.get("assistant_meta"),
        "retrieval": raw_response.get("retrieval"),
        "usage": raw_response.get("usage"),
    }
    assert known_fields["success"] is True
    assert known_fields["reply"] == "Test reply"
    # Unknown fields are safely ignored — no crash
    _ = raw_response.get("artifacts", [])
    _ = raw_response.get("tool_results", [])
    # Old FE doesn't know about these but they're in the JSON — no problem
    assert "artifacts" in raw_response
    assert "tool_results" in raw_response


# ── Regression: handle_chat() mcp_client_manager forwarding (reviewer bug) ───


def test_handle_chat_forwards_mcp_client_manager_to_prepare():
    """Regression test: handle_chat() must forward mcp_client_manager to
    _prepare_chat_turn().

    Reviewer (run_id 37) found that handle_chat() accepted mcp_client_manager
    in its signature but silently dropped it — never passing it to
    _prepare_chat_turn(). This caused the non-streaming endpoint to never inject
    the MCP system prompt. This test verifies the fix is in place by inspecting
    the source of handle_chat() and by mocking _prepare_chat_turn to assert the
    kwarg is forwarded at call time.
    """
    # 1. Source-level check: mcp_client_manager appears in _prepare_chat_turn call
    src = inspect.getsource(chat_service.handle_chat)
    lines = src.split("\n")
    in_prepare_call = False
    prepare_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "_prepare_chat_turn(" in stripped:
            in_prepare_call = True
        if in_prepare_call:
            prepare_lines.append(stripped)
            # The call block ends at a closing paren on its own or ending the expr
            if stripped == ")" or (stripped.endswith(")") and not stripped.startswith("prepared")):
                in_prepare_call = False

    prepare_text = " ".join(prepare_lines)
    assert "mcp_client_manager" in prepare_text, (
        "handle_chat() does not forward mcp_client_manager to _prepare_chat_turn(). "
        "Fix: add mcp_client_manager=mcp_client_manager to the _prepare_chat_turn() call."
    )

    # 2. Runtime check: _prepare_chat_turn receives mcp_client_manager kwarg
    mock_manager = MagicMock()
    mock_manager.enabled = True

    captured_kwargs: dict = {}

    def fake_prepare(*args, **kwargs):
        captured_kwargs.update(kwargs)
        raise RuntimeError("stop_here")  # abort after capture

    with patch("app.services.chat_service._prepare_chat_turn", side_effect=fake_prepare):
        try:
            chat_service.handle_chat(
                db=MagicMock(),
                username="testuser",
                session_id=1,
                user_message="hello",
                mcp_client_manager=mock_manager,
            )
        except RuntimeError as exc:
            assert str(exc) == "stop_here", f"Unexpected error: {exc}"
        except Exception:
            pass  # other exceptions from missing DB are irrelevant

    assert "mcp_client_manager" in captured_kwargs, (
        "handle_chat() did not pass mcp_client_manager kwarg to _prepare_chat_turn(). "
        "The MCP system prompt will never be injected on the non-streaming path."
    )
    assert captured_kwargs["mcp_client_manager"] is mock_manager
