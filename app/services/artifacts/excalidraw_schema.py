"""Validation and normalization for native Excalidraw artifacts."""
from __future__ import annotations

import hashlib
import json
from typing import Any


SUPPORTED_ELEMENT_TYPES = {
    "rectangle",
    "ellipse",
    "diamond",
    "text",
    "arrow",
    "line",
    "cameraUpdate",
}

DRAWABLE_ELEMENT_TYPES = SUPPORTED_ELEMENT_TYPES - {"cameraUpdate"}
DEFAULT_MAX_PAYLOAD_BYTES = 5 * 1024 * 1024


class ExcalidrawValidationError(ValueError):
    """Raised when model output cannot be safely rendered as Excalidraw data."""


def _stable_id(element: dict[str, Any], index: int) -> str:
    payload = json.dumps(element, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(f"{index}:{payload}".encode("utf-8")).hexdigest()[:16]
    return f"el_{digest}"


def _coerce_number(value: Any, *, default: float | None = None) -> float:
    if isinstance(value, bool):
        raise ExcalidrawValidationError("Boolean values are not valid coordinates.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError as exc:
            raise ExcalidrawValidationError(f"Invalid numeric value: {value!r}") from exc
    if default is not None:
        return float(default)
    raise ExcalidrawValidationError("Missing required numeric value.")


def _normalize_points(element: dict[str, Any]) -> list[list[float]]:
    points = element.get("points")
    if isinstance(points, list) and len(points) >= 2:
        normalized: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ExcalidrawValidationError("Line/arrow points must be [x, y] pairs.")
            normalized.append([_coerce_number(point[0]), _coerce_number(point[1])])
        return normalized

    if all(key in element for key in ("startX", "startY", "endX", "endY")):
        start_x = _coerce_number(element.get("startX"))
        start_y = _coerce_number(element.get("startY"))
        end_x = _coerce_number(element.get("endX"))
        end_y = _coerce_number(element.get("endY"))
        element["x"] = start_x
        element["y"] = start_y
        element["width"] = end_x - start_x
        element["height"] = end_y - start_y
        return [[0.0, 0.0], [end_x - start_x, end_y - start_y]]

    width = _coerce_number(element.get("width"), default=0)
    height = _coerce_number(element.get("height"), default=0)
    return [[0.0, 0.0], [width, height]]


def _base_defaults(element_type: str) -> dict[str, Any]:
    defaults = {
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if element_type in {"rectangle", "diamond"} else None,
        "seed": 1,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }
    if element_type == "text":
        defaults.update({
            "fontSize": 18,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": None,
            "lineHeight": 1.25,
        })
    if element_type == "arrow":
        defaults.update({"startArrowhead": None, "endArrowhead": "arrow"})
    return defaults


def normalize_excalidraw_elements(
    elements: Any,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> list[dict[str, Any]]:
    """Validate and normalize an Excalidraw element array.

    The function only accepts a small supported subset and never evaluates
    arbitrary content.
    """

    if not isinstance(elements, list):
        raise ExcalidrawValidationError("Excalidraw elements must be a JSON array.")

    try:
        payload_size = len(json.dumps(elements, ensure_ascii=False).encode("utf-8"))
    except TypeError as exc:
        raise ExcalidrawValidationError("Elements must be JSON serializable.") from exc
    if payload_size > max_payload_bytes:
        raise ExcalidrawValidationError("Excalidraw artifact exceeds configured size limit.")

    normalized_elements: list[dict[str, Any]] = []
    for index, raw_element in enumerate(elements):
        if not isinstance(raw_element, dict):
            raise ExcalidrawValidationError("Each Excalidraw element must be an object.")
        element_type = raw_element.get("type")
        if element_type not in SUPPORTED_ELEMENT_TYPES:
            raise ExcalidrawValidationError(f"Unsupported Excalidraw element type: {element_type!r}")

        element = {**_base_defaults(element_type), **raw_element}
        element["type"] = element_type
        element["x"] = _coerce_number(element.get("x"), default=0)
        element["y"] = _coerce_number(element.get("y"), default=0)
        element["width"] = _coerce_number(element.get("width"), default=0)
        element["height"] = _coerce_number(element.get("height"), default=0)

        if element_type == "cameraUpdate":
            normalized_elements.append({
                "type": "cameraUpdate",
                "x": element["x"],
                "y": element["y"],
                "width": max(1.0, abs(element["width"])),
                "height": max(1.0, abs(element["height"])),
            })
            continue

        element["id"] = str(element.get("id") or _stable_id(raw_element, index))
        if element_type == "text":
            text = element.get("text") or element.get("originalText")
            if not isinstance(text, str) or not text.strip():
                raise ExcalidrawValidationError("Text elements must contain text.")
            element["text"] = text
            element["originalText"] = element.get("originalText") or text

        if element_type in {"arrow", "line"}:
            element["points"] = _normalize_points(element)
            xs = [point[0] for point in element["points"]]
            ys = [point[1] for point in element["points"]]
            element["width"] = max(xs) - min(xs)
            element["height"] = max(ys) - min(ys)

        normalized_elements.append(element)

    return normalized_elements


def build_excalidraw_scene(elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "DominicChatbot",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},
        "files": {},
    }

