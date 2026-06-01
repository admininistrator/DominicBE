"""Small rule-based diagram intent detector.

The detector is deliberately isolated so it can be replaced by model-based
routing without changing the chat streaming path.
"""
from __future__ import annotations

import re
import unicodedata


_CLEAR_ENGLISH_PATTERNS = (
    re.compile(r"\bdraw\s+(?:an?\s+)?(?:architecture\s+)?diagram\b", re.IGNORECASE),
    re.compile(r"\b(?:architecture|system|sequence|uml|use\s+case)\s+diagram\b", re.IGNORECASE),
    re.compile(r"\b(?:flowchart|sequence\s+diagram|system\s+design\s+diagram)\b", re.IGNORECASE),
    re.compile(r"\bdraw\s+(?:a\s+)?(?:flow|flowchart|uml|sequence)\b", re.IGNORECASE),
)

_CLEAR_VIETNAMESE_PHRASES = (
    "ve so do",
    "ve flow",
    "ve uml",
    "ve sequence diagram",
    "ve kien truc",
    "ve so do kien truc",
    "so do luong",
    "so do kien truc",
)


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_prompt(value: str) -> str:
    normalized = _strip_accents(value).lower()
    normalized = normalized.replace("đ", "d").replace("Đ", "d")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def is_diagram_intent(user_message: str | None) -> bool:
    """Return True only for clear English/Vietnamese diagram requests."""

    raw = (user_message or "").strip()
    if not raw:
        return False
    if any(pattern.search(raw) for pattern in _CLEAR_ENGLISH_PATTERNS):
        return True

    normalized = _normalize_prompt(raw)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _CLEAR_VIETNAMESE_PHRASES):
        return True

    # Require both a drawing verb and a diagram noun to avoid routing broad
    # phrases like "draw conclusions" or "system design advice".
    has_draw_verb = bool(re.search(r"\b(?:draw|ve)\b", normalized))
    has_diagram_noun = bool(re.search(r"\b(?:diagram|flowchart|uml|flow|sequence|so do)\b", normalized))
    return has_draw_verb and has_diagram_noun

