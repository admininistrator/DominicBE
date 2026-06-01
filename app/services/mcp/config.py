"""Configuration models and registry loader for remote MCP servers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.services.mcp.exceptions import McpConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MCP_CONFIG_PATH = PROJECT_ROOT / "config" / "mcp_servers.json"


def _bool_from_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise McpConfigError(f"Invalid boolean MCP env override: {value!r}")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class McpServerConfig(BaseModel):
    id: str
    label: str
    url: str = ""
    enabled: bool = True
    auth_strategy: str | None = None
    auth_secret_env: str | None = None
    timeout_seconds: float = Field(default=30, gt=0)
    tool_allowlist: list[str] = Field(default_factory=list)
    artifact_capabilities: list[str] = Field(default_factory=list)
    health_check_interval_seconds: int = Field(default=300, ge=0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("id is required")
        return normalized

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("label is required")
        return normalized

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        return (value or "").strip().rstrip("/")

    @field_validator("auth_strategy")
    @classmethod
    def validate_auth_strategy(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized or normalized == "none":
            return None
        if normalized not in {"bearer", "api_key"}:
            raise ValueError("auth_strategy must be one of: bearer, api_key")
        return normalized

    @model_validator(mode="after")
    def validate_enabled_server(self):
        if self.enabled and not self.url:
            raise ValueError("enabled MCP server requires url")
        return self

    @property
    def env_prefix(self) -> str:
        return "MCP_SERVER_" + "".join(ch if ch.isalnum() else "_" for ch in self.id.upper())

    def is_tool_allowed(self, tool_name: str) -> bool:
        if not self.tool_allowlist:
            return True
        return tool_name in self.tool_allowlist


class McpGlobalConfig(BaseModel):
    enabled: bool = False
    remote_enabled: bool = True
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0)
    tool_invocation_enabled: bool = True
    artifact_storage_mode: str = "inline"
    total_budget_seconds: float = Field(default=60, gt=0)
    tool_cache_ttl_seconds: float = Field(default=300, ge=0)
    max_artifact_content_bytes: int = Field(default=500 * 1024, ge=1024)

    @classmethod
    def from_settings(cls, settings: Any) -> "McpGlobalConfig":
        return cls(
            enabled=bool(getattr(settings, "mcp_enabled", False)),
            remote_enabled=bool(getattr(settings, "mcp_remote_enabled", True)),
            timeout_seconds=float(getattr(settings, "mcp_timeout_seconds", 30)),
            max_retries=int(getattr(settings, "mcp_max_retries", 2)),
            tool_invocation_enabled=bool(getattr(settings, "mcp_tool_invocation_enabled", True)),
            artifact_storage_mode=str(getattr(settings, "mcp_artifact_storage_mode", "inline") or "inline"),
            total_budget_seconds=float(getattr(settings, "mcp_total_budget_seconds", 60)),
            tool_cache_ttl_seconds=float(getattr(settings, "mcp_tool_cache_ttl_seconds", 300)),
            max_artifact_content_bytes=int(getattr(settings, "mcp_max_artifact_content_bytes", 500 * 1024)),
        )


class McpConfig(BaseModel):
    servers: list[McpServerConfig] = Field(default_factory=list)
    global_config: McpGlobalConfig = Field(default_factory=McpGlobalConfig)

    def get_server(self, server_id: str) -> McpServerConfig | None:
        return next((server for server in self.servers if server.id == server_id), None)


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return DEFAULT_MCP_CONFIG_PATH
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _apply_server_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    server = McpServerConfig.model_construct(**{**data, "id": str(data.get("id", "")), "label": str(data.get("label", data.get("id", "")))})
    prefix = server.env_prefix
    overridden = dict(data)

    if f"{prefix}_ENABLED" in env:
        overridden["enabled"] = _bool_from_env(env[f"{prefix}_ENABLED"])
    if f"{prefix}_URL" in env:
        overridden["url"] = env[f"{prefix}_URL"]
    if f"{prefix}_TIMEOUT" in env:
        overridden["timeout_seconds"] = float(env[f"{prefix}_TIMEOUT"])
    if f"{prefix}_TOOL_ALLOWLIST" in env:
        overridden["tool_allowlist"] = _split_csv(env[f"{prefix}_TOOL_ALLOWLIST"])
    if f"{prefix}_ARTIFACT_CAPABILITIES" in env:
        overridden["artifact_capabilities"] = _split_csv(env[f"{prefix}_ARTIFACT_CAPABILITIES"])
    if f"{prefix}_TAGS" in env:
        overridden["tags"] = _split_csv(env[f"{prefix}_TAGS"])
    if f"{prefix}_AUTH_STRATEGY" in env:
        value = env[f"{prefix}_AUTH_STRATEGY"].strip()
        overridden["auth_strategy"] = value or None
    if f"{prefix}_AUTH_SECRET_ENV" in env:
        value = env[f"{prefix}_AUTH_SECRET_ENV"].strip()
        overridden["auth_secret_env"] = value or None

    return overridden


def load_mcp_config(
    *,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    global_config: McpGlobalConfig | None = None,
) -> McpConfig:
    """Load MCP server registry JSON and apply environment overrides.

    Disabled servers are excluded from the returned config. Enabled servers with
    missing URLs are rejected so user-controlled or malformed MCP endpoints never
    enter the runtime registry.
    """

    path = _resolve_config_path(config_path)
    environment = os.environ if env is None else env
    if not path.exists():
        return McpConfig(servers=[], global_config=global_config or McpGlobalConfig())

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"MCP registry contains invalid JSON: line {exc.lineno} column {exc.colno}") from exc

    raw_servers = raw.get("servers", [])
    if not isinstance(raw_servers, list):
        raise McpConfigError("MCP registry 'servers' must be a list")

    servers: list[McpServerConfig] = []
    for item in raw_servers:
        if not isinstance(item, dict):
            raise McpConfigError("Each MCP server entry must be an object")
        data = _apply_server_env_overrides(item, environment)
        try:
            server = McpServerConfig(**data)
        except ValidationError as exc:
            raise McpConfigError(f"Invalid MCP server {item.get('id', '<unknown>')}: {exc.errors()[0]['msg']}") from exc
        if server.enabled:
            servers.append(server)

    return McpConfig(servers=servers, global_config=global_config or McpGlobalConfig())
