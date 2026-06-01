from __future__ import annotations

import json

import pytest

from app.services.mcp.config import McpConfigError, McpGlobalConfig, load_mcp_config


def _write_registry(tmp_path, servers):
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return config_path


def _server(**overrides):
    data = {
        "id": "excalidraw",
        "label": "Excalidraw Whiteboard",
        "url": "https://mcp.excalidraw.com",
        "enabled": True,
        "auth_strategy": None,
        "auth_secret_env": None,
        "timeout_seconds": 30,
        "tool_allowlist": [],
        "artifact_capabilities": ["excalidraw_json", "link"],
        "health_check_interval_seconds": 300,
        "tags": ["drawing"],
    }
    data.update(overrides)
    return data


def test_load_mcp_config_filters_disabled_servers(tmp_path):
    config_path = _write_registry(
        tmp_path,
        [
            _server(id="enabled", url="https://enabled.example/mcp"),
            _server(id="disabled", url="https://disabled.example/mcp", enabled=False),
        ],
    )

    config = load_mcp_config(config_path=config_path, env={})

    assert [server.id for server in config.servers] == ["enabled"]
    assert config.servers[0].url == "https://enabled.example/mcp"


def test_load_mcp_config_rejects_enabled_server_missing_url(tmp_path):
    config_path = _write_registry(tmp_path, [_server(url="")])

    with pytest.raises(McpConfigError, match="url"):
        load_mcp_config(config_path=config_path, env={})


def test_load_mcp_config_applies_server_env_overrides(tmp_path):
    config_path = _write_registry(tmp_path, [_server(timeout_seconds=30, enabled=True)])

    config = load_mcp_config(
        config_path=config_path,
        env={
            "MCP_SERVER_EXCALIDRAW_ENABLED": "false",
            "MCP_SERVER_EXCALIDRAW_URL": "https://override.example/mcp",
            "MCP_SERVER_EXCALIDRAW_TIMEOUT": "7",
            "MCP_SERVER_EXCALIDRAW_TOOL_ALLOWLIST": "draw, export_png",
        },
    )

    assert config.servers == []

    config = load_mcp_config(
        config_path=config_path,
        env={
            "MCP_SERVER_EXCALIDRAW_ENABLED": "true",
            "MCP_SERVER_EXCALIDRAW_URL": "https://override.example/mcp",
            "MCP_SERVER_EXCALIDRAW_TIMEOUT": "7",
            "MCP_SERVER_EXCALIDRAW_TOOL_ALLOWLIST": "draw, export_png",
        },
    )

    server = config.servers[0]
    assert server.url == "https://override.example/mcp"
    assert server.timeout_seconds == 7
    assert server.tool_allowlist == ["draw", "export_png"]


def test_global_config_reads_settings_fields():
    class FakeSettings:
        mcp_enabled = True
        mcp_remote_enabled = False
        mcp_timeout_seconds = 12
        mcp_max_retries = 3
        mcp_tool_invocation_enabled = True
        mcp_artifact_storage_mode = "inline"
        mcp_total_budget_seconds = 45
        mcp_tool_cache_ttl_seconds = 111

    config = McpGlobalConfig.from_settings(FakeSettings())

    assert config.enabled is True
    assert config.remote_enabled is False
    assert config.timeout_seconds == 12
    assert config.max_retries == 3
    assert config.tool_invocation_enabled is True
    assert config.artifact_storage_mode == "inline"
    assert config.total_budget_seconds == 45
    assert config.tool_cache_ttl_seconds == 111


def test_missing_registry_file_loads_empty_config(tmp_path):
    config = load_mcp_config(config_path=tmp_path / "missing.json", env={})

    assert config.servers == []
