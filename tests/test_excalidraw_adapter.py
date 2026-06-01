from __future__ import annotations

import json

from app.services.mcp.adapters import GenericResultAdapter, get_result_adapter
from app.services.mcp.adapters.excalidraw import ExcalidrawResultAdapter


def test_adapter_registry_returns_excalidraw_specific_adapter():
    adapter = get_result_adapter("excalidraw", tool_name="create-excalidraw")

    assert isinstance(adapter, ExcalidrawResultAdapter)
    assert isinstance(get_result_adapter("other-server"), GenericResultAdapter)


def test_excalidraw_link_output_normalizes_to_safe_excalidraw_artifact():
    adapter = ExcalidrawResultAdapter(tool_name="create-excalidraw")

    artifacts = adapter.normalize({"content": [{"type": "text", "text": "https://excalidraw.com/#json=abc"}]})

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "excalidraw"
    assert artifact.url == "https://excalidraw.com/#json=abc"
    assert artifact.content is None
    assert artifact.tool_server_id == "excalidraw"
    assert artifact.tool_name == "create-excalidraw"
    assert artifact.safe is True


def test_sdk_call_tool_result_like_object_with_text_content_link_normalizes_to_artifact():
    class TextContentLike:
        type = "text"
        text = "https://excalidraw.com/#json=sdk"

    class CallToolResultLike:
        content = [TextContentLike()]

    adapter = ExcalidrawResultAdapter(tool_name="export_to_excalidraw")

    artifacts = adapter.normalize(CallToolResultLike())

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "excalidraw"
    assert artifact.url == "https://excalidraw.com/#json=sdk"
    assert artifact.content is None
    assert artifact.safe is True


def test_sdk_object_with_json_scene_text_normalizes_to_content_artifact():
    scene = {"type": "excalidraw", "version": 2, "elements": []}

    class TextContentLike:
        def model_dump(self):
            return {"type": "text", "text": json.dumps(scene)}

    class CallToolResultLike:
        def model_dump(self):
            return {"content": [TextContentLike()]}

    adapter = ExcalidrawResultAdapter(tool_name="export_to_excalidraw")

    artifacts = adapter.normalize(CallToolResultLike())

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "excalidraw"
    assert artifact.mime_type == "application/json"
    assert json.loads(artifact.content) == scene
    assert artifact.safe is True


def test_excalidraw_json_output_normalizes_to_safe_json_artifact():
    adapter = ExcalidrawResultAdapter(tool_name="create-excalidraw")
    scene = {"type": "excalidraw", "version": 2, "elements": []}

    artifacts = adapter.normalize({"content": [{"type": "text", "text": json.dumps(scene)}]})

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "excalidraw"
    assert artifact.mime_type == "application/json"
    assert json.loads(artifact.content) == scene
    assert artifact.safe is True


def test_excalidraw_image_url_output_normalizes_to_safe_image_artifact():
    adapter = ExcalidrawResultAdapter(tool_name="export-image")

    artifacts = adapter.normalize(
        {
            "content": [
                {
                    "type": "image",
                    "url": "https://cdn.excalidraw.com/export.png",
                    "mimeType": "image/png",
                }
            ]
        }
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "image"
    assert artifact.url == "https://cdn.excalidraw.com/export.png"
    assert artifact.mime_type == "image/png"
    assert artifact.safe is True


def test_excalidraw_adapter_rejects_unsafe_urls():
    adapter = ExcalidrawResultAdapter(tool_name="create-excalidraw")

    assert adapter.normalize({"content": [{"type": "text", "text": "javascript:alert(1)"}]}) == []
    assert adapter.normalize({"content": [{"type": "text", "text": "file:///c:/secret.txt"}]}) == []
    assert adapter.normalize({"content": [{"type": "text", "text": "http://insecure.example/result"}]}) == []


def test_excalidraw_svg_content_is_sanitized_before_safe_flag():
    adapter = ExcalidrawResultAdapter(tool_name="export-image")
    malicious_svg = '<svg onload="alert(1)"><script>alert(1)</script><rect onclick="bad()" /></svg>'

    artifacts = adapter.normalize({"content": [{"type": "text", "mimeType": "image/svg+xml", "text": malicious_svg}]})

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "image"
    assert artifact.mime_type == "image/svg+xml"
    assert "<script" not in artifact.content.lower()
    assert "onload" not in artifact.content.lower()
    assert "onclick" not in artifact.content.lower()
    assert artifact.safe is True


def test_oversized_excalidraw_json_drops_inline_content_but_keeps_safe_link():
    adapter = ExcalidrawResultAdapter(tool_name="create-excalidraw", max_content_bytes=32)
    raw = {
        "content": [
            {
                "type": "json",
                "json": {"type": "excalidraw", "elements": ["x" * 128]},
                "url": "https://excalidraw.com/#json=abc",
            }
        ]
    }

    artifacts = adapter.normalize(raw)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "excalidraw"
    assert artifact.url == "https://excalidraw.com/#json=abc"
    assert artifact.content is None
    assert artifact.safe is True
    assert artifact.metadata["content_dropped_reason"] == "size_limit_exceeded"


def test_malformed_excalidraw_response_is_safe_empty_result():
    adapter = ExcalidrawResultAdapter(tool_name="create-excalidraw")

    assert adapter.normalize({"content": [None, {"type": "text"}, object()]}) == []
