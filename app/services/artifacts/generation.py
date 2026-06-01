"""Generic artifact generation provider interfaces and selection helpers.

The chat service owns provider-specific prompt/streaming mechanics; this module
keeps provider choice explicit and testable so future artifact kinds can plug in
without changing the SSE protocol.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ArtifactGenerationRequest:
    """Inputs passed to an artifact provider after model generation."""

    kind: str
    user_message: str
    request_id: str
    artifact_id: str
    title: str
    model_output: str = ""
    elements: list[dict[str, Any]] | None = None
    username: str | None = None
    session_id: int | None = None
    context: Any = None


@dataclass
class ArtifactGenerationResult:
    """Normalized result from one artifact provider."""

    provider_id: str
    artifacts: list[Any] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    fallback_used: bool = False
    error: str | None = None

    @property
    def has_artifacts(self) -> bool:
        return bool(self.artifacts)


class ArtifactProvider(Protocol):
    """Provider contract for a future-proof artifact generation service."""

    provider_id: str
    kind: str

    def is_available(self, request: ArtifactGenerationRequest) -> bool:
        """Return True when this provider can service the request."""

    def generate(self, request: ArtifactGenerationRequest) -> ArtifactGenerationResult:
        """Generate artifacts from an already-prepared model output."""


AvailabilityPredicate = Callable[[ArtifactGenerationRequest], bool]
GenerationCallable = Callable[[ArtifactGenerationRequest], ArtifactGenerationResult]


class CallableArtifactProvider:
    """Small adapter for providers implemented by existing project functions."""

    def __init__(
        self,
        *,
        provider_id: str,
        kind: str,
        generate: GenerationCallable,
        is_available: AvailabilityPredicate | bool = True,
    ):
        self.provider_id = provider_id
        self.kind = kind
        self._generate = generate
        self._is_available = is_available

    def is_available(self, request: ArtifactGenerationRequest) -> bool:
        if callable(self._is_available):
            return bool(self._is_available(request))
        return bool(self._is_available)

    def generate(self, request: ArtifactGenerationRequest) -> ArtifactGenerationResult:
        result = self._generate(request)
        if not isinstance(result, ArtifactGenerationResult):
            raise TypeError("Artifact provider returned an invalid result object")
        if not result.provider_id:
            result.provider_id = self.provider_id
        return result


class ArtifactGenerationService:
    """Select and run artifact providers with an optional native fallback."""

    def __init__(
        self,
        providers: list[ArtifactProvider] | tuple[ArtifactProvider, ...],
        *,
        fallback_provider: ArtifactProvider | None = None,
    ):
        self.providers = list(providers)
        self.fallback_provider = fallback_provider

    def select_provider(self, request: ArtifactGenerationRequest) -> ArtifactProvider | None:
        for provider in self.providers:
            if provider.kind == request.kind and provider.is_available(request):
                return provider
        if (
            self.fallback_provider is not None
            and self.fallback_provider.kind == request.kind
            and self.fallback_provider.is_available(request)
        ):
            return self.fallback_provider
        return None

    def generate(self, request: ArtifactGenerationRequest) -> ArtifactGenerationResult:
        provider = self.select_provider(request)
        if provider is None:
            return ArtifactGenerationResult(provider_id="none")

        result = provider.generate(request)
        if result.has_artifacts or self.fallback_provider is None or provider is self.fallback_provider:
            return result

        if self.fallback_provider.kind != request.kind or not self.fallback_provider.is_available(request):
            return result

        fallback = self.fallback_provider.generate(request)
        fallback.fallback_used = True
        # Preserve attempted tool results for observability when falling back.
        fallback.tool_results = [*result.tool_results, *fallback.tool_results]
        if fallback.error is None:
            fallback.error = result.error
        return fallback


def _manager_tool_allowed(mcp_client_manager: Any, server_id: str, tool_name: str) -> bool:
    checker = getattr(mcp_client_manager, "is_tool_allowed", None)
    if callable(checker):
        try:
            return bool(checker(server_id, tool_name))
        except Exception:  # noqa: BLE001 - provider selection must fail closed
            return False

    config = getattr(mcp_client_manager, "config", None)
    server = config.get_server(server_id) if config is not None and hasattr(config, "get_server") else None
    if server is None:
        return False
    is_tool_allowed = getattr(server, "is_tool_allowed", None)
    return bool(is_tool_allowed(tool_name)) if callable(is_tool_allowed) else True


def is_excalidraw_mcp_provider_available(
    mcp_client_manager: Any,
    *,
    server_id: str = "excalidraw",
    create_view_tool: str = "create_view",
    export_tool: str = "export_to_excalidraw",
    provider_mode: str = "auto",
) -> bool:
    """Return True only for a compatible configured Excalidraw MCP provider.

    This intentionally checks configuration and allowlists without doing network
    discovery on every chat turn. Actual remote tool failures still fall back to
    safe inline/native artifacts at invocation time.
    """

    mode = (provider_mode or "auto").strip().lower()
    if mode == "native":
        return False
    if not mcp_client_manager or not getattr(mcp_client_manager, "enabled", False):
        return False

    config = getattr(mcp_client_manager, "config", None)
    if config is None or not hasattr(config, "get_server"):
        return False
    server = config.get_server(server_id)
    if server is None or getattr(server, "enabled", True) is False:
        return False

    capabilities = [str(item).strip().lower() for item in (getattr(server, "artifact_capabilities", []) or [])]
    tags = [str(item).strip().lower() for item in (getattr(server, "tags", []) or [])]
    compatibility_tokens = [*capabilities, *tags, str(getattr(server, "id", "")).lower()]
    if compatibility_tokens and not any(
        "excalidraw" in token or "diagram" in token or "whiteboard" in token or token in {"mcp_app", "mcp-app"}
        for token in compatibility_tokens
    ):
        return False

    return any(
        _manager_tool_allowed(mcp_client_manager, server_id, tool_name)
        for tool_name in (create_view_tool, export_tool)
        if tool_name
    )


def selected_excalidraw_provider_id(
    *,
    diagram_intent: bool,
    mcp_client_manager: Any,
    server_id: str = "excalidraw",
    create_view_tool: str = "create_view",
    export_tool: str = "export_to_excalidraw",
    provider_mode: str = "auto",
) -> str:
    """Resolve the Excalidraw provider id for diagnostics/tests."""

    if not diagram_intent:
        return "none"
    if is_excalidraw_mcp_provider_available(
        mcp_client_manager,
        server_id=server_id,
        create_view_tool=create_view_tool,
        export_tool=export_tool,
        provider_mode=provider_mode,
    ):
        return "mcp_excalidraw"
    return "native_llm_excalidraw"
