"""Streaming helpers for native Excalidraw artifacts."""
from __future__ import annotations

import json
import re
from typing import Any

from app.services.artifacts.excalidraw_schema import (
    ExcalidrawValidationError,
    build_excalidraw_scene,
    normalize_excalidraw_elements,
)

JSON_FENCE_PATTERN = re.compile(r"```(?:json|excalidraw)?\s*([\s\S]*?)```", re.IGNORECASE)


def build_diagram_system_prompt() -> str:
    return "\n".join(
        [
            "You create editable Excalidraw-compatible diagram artifacts.",
            "Return only a compact JSON array. Do not use markdown, prose, or code fences.",
            "Every item must be an object with a supported type: cameraUpdate, rectangle, ellipse, diamond, text, arrow, line.",
            "The first item must be a cameraUpdate pseudo-element with x, y, width, and height.",
            "Drawable elements require id, type, x, y, width, and height.",
            "Text elements require text. Arrows and lines should include x, y, width, height, and points such as [[0,0],[180,0]].",
            "Use stream-friendly drawing order: cameraUpdate, background/container, node, label, arrow, next node, label, arrow.",
            "Keep labels concise and make the diagram readable on a white background.",
        ]
    )


def json_candidates_from_text(text: str) -> list[Any]:
    candidates: list[Any] = []
    for match in JSON_FENCE_PATTERN.finditer(text or ""):
        fenced = match.group(1).strip()
        if not fenced:
            continue
        try:
            candidates.append(json.loads(fenced))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(parsed)
    return candidates


def partial_json_array_objects(text: str) -> list[dict[str, Any]]:
    """Extract complete objects from a still-incomplete JSON array string."""

    raw = text or ""
    elements_index = raw.find('"elements"')
    array_start = raw.find("[", elements_index if elements_index >= 0 else 0)
    if array_start < 0:
        return []

    objects: list[dict[str, Any]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index in range(array_start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(raw[start:index + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    objects.append(parsed)
                start = None
    return objects


def elements_from_candidate(candidate: Any) -> list[dict[str, Any]] | None:
    if isinstance(candidate, list):
        return candidate if all(isinstance(item, dict) for item in candidate) else None
    if isinstance(candidate, dict) and isinstance(candidate.get("elements"), list):
        raw_elements = candidate["elements"]
        return raw_elements if all(isinstance(item, dict) for item in raw_elements) else None
    return None


def parse_final_elements(text: str, *, max_payload_bytes: int) -> list[dict[str, Any]]:
    for candidate in json_candidates_from_text(text):
        raw_elements = elements_from_candidate(candidate)
        if raw_elements is None:
            continue
        return normalize_excalidraw_elements(raw_elements, max_payload_bytes=max_payload_bytes)
    raise ExcalidrawValidationError("No valid Excalidraw JSON array was found.")


def repair_final_elements(text: str, *, max_payload_bytes: int) -> list[dict[str, Any]]:
    partial = partial_json_array_objects(text)
    if not partial:
        raise ExcalidrawValidationError("Unable to repair Excalidraw JSON output.")
    return normalize_excalidraw_elements(partial, max_payload_bytes=max_payload_bytes)


def artifact_id_for_request(request_id: str) -> str:
    return f"excalidraw_{request_id}"


def artifact_response_from_elements(
    elements: list[dict[str, Any]],
    *,
    artifact_id: str,
    title: str,
    request_id: str,
    streaming: bool,
    sequence: int | None = None,
) -> dict[str, Any]:
    scene = build_excalidraw_scene(elements)
    metadata: dict[str, Any] = {
        "source": "llm",
        "format": "excalidraw-elements-v1",
        "request_id": request_id,
        "render_mode": "native_excalidraw_stream",
        "streaming": streaming,
        "element_count": len(elements),
    }
    if sequence is not None:
        metadata["sequence"] = sequence
    return {
        "id": artifact_id,
        "type": "excalidraw",
        "title": title,
        "content": json.dumps(scene, ensure_ascii=False, separators=(",", ":")),
        "url": None,
        "preview_url": None,
        "metadata": metadata,
    }


def artifact_start_event(*, artifact_id: str, title: str, request_id: str) -> dict[str, Any]:
    return {
        "type": "artifact_start",
        "request_id": request_id,
        "artifact": {
            "id": artifact_id,
            "kind": "excalidraw",
            "mode": "streaming",
            "title": title,
        },
    }


def artifact_delta_event(
    *,
    artifact_id: str,
    title: str,
    request_id: str,
    elements_partial: str,
    elements: list[dict[str, Any]],
    sequence: int,
) -> dict[str, Any]:
    return {
        "type": "artifact_delta",
        "request_id": request_id,
        "artifactId": artifact_id,
        "kind": "excalidraw",
        "elementsPartial": elements_partial,
        "artifact": artifact_response_from_elements(
            elements,
            artifact_id=artifact_id,
            title=title,
            request_id=request_id,
            streaming=True,
            sequence=sequence,
        ),
    }


def artifact_done_event(
    *,
    artifact_id: str,
    title: str,
    request_id: str,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "artifact_done",
        "request_id": request_id,
        "artifactId": artifact_id,
        "kind": "excalidraw",
        "elements": elements,
        "metadata": {"source": "llm", "format": "excalidraw-elements-v1"},
        "artifact": artifact_response_from_elements(
            elements,
            artifact_id=artifact_id,
            title=title,
            request_id=request_id,
            streaming=False,
        ),
    }


def artifact_error_event(*, artifact_id: str, request_id: str, message: str) -> dict[str, Any]:
    return {
        "type": "artifact_error",
        "request_id": request_id,
        "artifactId": artifact_id,
        "kind": "excalidraw",
        "message": message,
    }

