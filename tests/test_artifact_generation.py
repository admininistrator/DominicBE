from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.artifacts.generation import (
    ArtifactGenerationRequest,
    ArtifactGenerationResult,
    ArtifactGenerationService,
    CallableArtifactProvider,
    is_excalidraw_mcp_provider_available,
    selected_excalidraw_provider_id,
)


def _request(**overrides: Any) -> ArtifactGenerationRequest:
    data = {
        "kind": "excalidraw",
        "user_message": "draw an architecture diagram",
        "request_id": "req-1",
        "artifact_id": "excalidraw_req-1",
        "title": "Diagram",
        "model_output": "[]",
        "elements": [{"type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 50}],
    }
    data.update(overrides)
    return ArtifactGenerationRequest(**data)


def test_artifact_generation_service_selects_available_provider_before_fallback():
    calls: list[str] = []
    request = _request()

    mcp_provider = CallableArtifactProvider(
        provider_id="mcp_excalidraw",
        kind="excalidraw",
        is_available=True,
        generate=lambda _request: calls.append("mcp") or ArtifactGenerationResult(
            provider_id="mcp_excalidraw",
            artifacts=[{"id": "mcp-art"}],
        ),
    )
    native_provider = CallableArtifactProvider(
        provider_id="native_llm_excalidraw",
        kind="excalidraw",
        is_available=True,
        generate=lambda _request: calls.append("native") or ArtifactGenerationResult(
            provider_id="native_llm_excalidraw",
            artifacts=[{"id": "native-art"}],
        ),
    )

    result = ArtifactGenerationService([mcp_provider], fallback_provider=native_provider).generate(request)

    assert result.provider_id == "mcp_excalidraw"
    assert result.artifacts == [{"id": "mcp-art"}]
    assert result.fallback_used is False
    assert calls == ["mcp"]


def test_artifact_generation_service_falls_back_when_primary_returns_no_artifacts():
    request = _request()
    mcp_provider = CallableArtifactProvider(
        provider_id="mcp_excalidraw",
        kind="excalidraw",
        is_available=True,
        generate=lambda _request: ArtifactGenerationResult(
            provider_id="mcp_excalidraw",
            artifacts=[],
            tool_results=[{"status": "connection_error"}],
            error="remote failed",
        ),
    )
    native_provider = CallableArtifactProvider(
        provider_id="native_llm_excalidraw",
        kind="excalidraw",
        is_available=lambda provider_request: bool(provider_request.elements),
        generate=lambda _request: ArtifactGenerationResult(
            provider_id="native_llm_excalidraw",
            artifacts=[{"id": "native-art"}],
        ),
    )

    result = ArtifactGenerationService([mcp_provider], fallback_provider=native_provider).generate(request)

    assert result.provider_id == "native_llm_excalidraw"
    assert result.artifacts == [{"id": "native-art"}]
    assert result.tool_results == [{"status": "connection_error"}]
    assert result.fallback_used is True
    assert result.error == "remote failed"


def test_artifact_generation_service_returns_none_result_when_no_provider_available():
    request = _request(elements=None)
    unavailable = CallableArtifactProvider(
        provider_id="mcp_excalidraw",
        kind="excalidraw",
        is_available=False,
        generate=lambda _request: ArtifactGenerationResult(provider_id="mcp_excalidraw"),
    )
    native_provider = CallableArtifactProvider(
        provider_id="native_llm_excalidraw",
        kind="excalidraw",
        is_available=lambda provider_request: bool(provider_request.elements),
        generate=lambda _request: ArtifactGenerationResult(provider_id="native_llm_excalidraw"),
    )

    result = ArtifactGenerationService([unavailable], fallback_provider=native_provider).generate(request)

    assert result.provider_id == "none"
    assert result.artifacts == []
    assert result.tool_results == []


@dataclass
class FakeServer:
    id: str = "excalidraw"
    enabled: bool = True
    artifact_capabilities: list[str] = field(default_factory=lambda: ["excalidraw_json"])
    tags: list[str] = field(default_factory=list)
    tool_allowlist: list[str] = field(default_factory=list)

    def is_tool_allowed(self, tool_name: str) -> bool:
        return not self.tool_allowlist or tool_name in self.tool_allowlist


class FakeConfig:
    def __init__(self, server: FakeServer | None):
        self.server = server

    def get_server(self, server_id: str):
        if self.server and self.server.id == server_id:
            return self.server
        return None


class FakeManager:
    def __init__(self, server: FakeServer | None, *, enabled: bool = True, allowed_tools: set[str] | None = None):
        self.enabled = enabled
        self.config = FakeConfig(server)
        self.allowed_tools = allowed_tools

    def is_tool_allowed(self, server_id: str, tool_name: str) -> bool:
        server = self.config.get_server(server_id)
        if server is None:
            return False
        if self.allowed_tools is not None:
            return tool_name in self.allowed_tools
        return server.is_tool_allowed(tool_name)


def test_excalidraw_mcp_provider_available_in_auto_and_mcp_modes_when_configured():
    manager = FakeManager(FakeServer(tool_allowlist=["create_view"]), allowed_tools={"create_view"})

    assert is_excalidraw_mcp_provider_available(manager, provider_mode="auto") is True
    assert is_excalidraw_mcp_provider_available(manager, provider_mode="mcp") is True
    assert selected_excalidraw_provider_id(diagram_intent=True, mcp_client_manager=manager) == "mcp_excalidraw"
    assert selected_excalidraw_provider_id(diagram_intent=False, mcp_client_manager=manager) == "none"


def test_excalidraw_mcp_provider_native_mode_forces_native_selection():
    manager = FakeManager(FakeServer(tool_allowlist=["create_view"]), allowed_tools={"create_view"})

    assert is_excalidraw_mcp_provider_available(manager, provider_mode="native") is False
    assert selected_excalidraw_provider_id(
        diagram_intent=True,
        mcp_client_manager=manager,
        provider_mode="native",
    ) == "native_llm_excalidraw"


def test_excalidraw_mcp_provider_fails_closed_for_missing_disabled_or_incompatible_config():
    assert is_excalidraw_mcp_provider_available(None) is False
    assert is_excalidraw_mcp_provider_available(FakeManager(None)) is False
    assert is_excalidraw_mcp_provider_available(FakeManager(FakeServer(enabled=False))) is False
    assert is_excalidraw_mcp_provider_available(FakeManager(FakeServer(), enabled=False)) is False
    assert is_excalidraw_mcp_provider_available(
        FakeManager(FakeServer(id="generic", artifact_capabilities=["chart"], tags=["analytics"]))
    ) is False


def test_excalidraw_mcp_provider_respects_tool_allowlist_and_custom_server_id():
    denied = FakeManager(FakeServer(tool_allowlist=["other_tool"]), allowed_tools={"other_tool"})
    assert is_excalidraw_mcp_provider_available(denied) is False
    assert selected_excalidraw_provider_id(diagram_intent=True, mcp_client_manager=denied) == "native_llm_excalidraw"

    custom_server = FakeServer(
        id="whiteboard-prod",
        artifact_capabilities=["mcp_app"],
        tags=["whiteboard"],
        tool_allowlist=["render_app"],
    )
    custom_manager = FakeManager(custom_server, allowed_tools={"render_app"})

    assert is_excalidraw_mcp_provider_available(
        custom_manager,
        server_id="whiteboard-prod",
        create_view_tool="render_app",
        export_tool="",
    ) is True
    assert selected_excalidraw_provider_id(
        diagram_intent=True,
        mcp_client_manager=custom_manager,
        server_id="whiteboard-prod",
        create_view_tool="render_app",
        export_tool="",
    ) == "mcp_excalidraw"
