import json
from typing import Any


def normalize_json_value(value: Any, *, max_depth: int = 2) -> Any:
    current = value
    for _ in range(max(0, max_depth)):
        if not isinstance(current, str):
            break
        text = current.strip()
        if not text:
            break
        try:
            current = json.loads(text)
        except Exception:
            break
    return current


def ensure_json_mapping(value: Any) -> dict:
    normalized = normalize_json_value(value)
    return normalized if isinstance(normalized, dict) else {}


def ensure_json_list(value: Any) -> list:
    normalized = normalize_json_value(value)
    return normalized if isinstance(normalized, list) else []