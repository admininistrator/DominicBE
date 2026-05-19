"""Qwen3 embedding smoke probe (P02-T02).

Verifies that the Ollama server is reachable, that qwen3-embedding:0.6b (or the
configured model) is available, and that a minimal embed call returns a valid
vector with the expected dimensions.

Usage
-----
Run directly from the DominicBE project root:

    python scripts/qwen_smoke_probe.py

Or with an explicit base URL / model override:

    EMBEDDING_BASE_URL=http://host.docker.internal:11434 \\
    EMBEDDING_MODEL=qwen3-embedding:0.6b \\
    python scripts/qwen_smoke_probe.py

Exit codes
----------
0  All checks passed.
1  Ollama unavailable, model missing, or embedding validation failed.

Design constraints
------------------
- Does NOT require ingestion, a running FastAPI server, or a database.
- Does NOT log raw document text.
- Reports: model, dimensions, latency_ms, base_url, and failure category.
- Uses httpx (already in requirements.txt) for HTTP calls.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import httpx
except ImportError:
    print("ERROR: httpx is not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration — read from environment / settings with safe fallback
# ---------------------------------------------------------------------------
try:
    from app.core.config import settings as _settings

    _BASE_URL: str = (_settings.embedding_base_url or "http://localhost:11434").rstrip("/")
    _MODEL: str = _settings.embedding_model or "qwen3-embedding:0.6b"
    _TIMEOUT: float = _settings.embedding_timeout_seconds or 60.0
    _EXPECTED_DIMS: int = _settings.embedding_dimensions or 0
except Exception:
    import os

    _BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:11434").rstrip("/")
    _MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    _TIMEOUT = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60"))
    _EXPECTED_DIMS = int(os.environ.get("EMBEDDING_DIMENSIONS", "0"))

_TAGS_URL = f"{_BASE_URL}/api/tags"
_EMBED_URL = f"{_BASE_URL}/api/embed"

# Probe text — short, generic, no sensitive content
_PROBE_TEXT = "embedding smoke probe"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _print_result(label: str, value: object) -> None:
    print(f"  {label:<22} {value}")


def _fail(message: str, *, category: str = "unknown") -> None:
    print(f"\nFAIL [{category}]: {message}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Check Ollama tags endpoint (model discovery)
# ---------------------------------------------------------------------------

def check_ollama_tags() -> list[str]:
    """GET /api/tags and return list of available model names."""
    print(f"\n[1/3] Checking Ollama availability at {_BASE_URL} ...")
    try:
        resp = httpx.get(_TAGS_URL, timeout=_TIMEOUT)
    except httpx.ConnectError as exc:
        _fail(
            f"Cannot connect to Ollama at {_BASE_URL}. "
            "Check EMBEDDING_BASE_URL and that Ollama is running.\n"
            f"  Detail: {exc}",
            category="connection_error",
        )
    except httpx.TimeoutException:
        _fail(
            f"Connection to {_TAGS_URL} timed out after {_TIMEOUT}s.",
            category="timeout",
        )
    except httpx.RequestError as exc:
        _fail(f"HTTP request error: {exc}", category="request_error")

    if not resp.is_success:
        _fail(
            f"Ollama /api/tags returned HTTP {resp.status_code}.",
            category="http_error",
        )

    try:
        data = resp.json()
    except Exception:
        _fail("Ollama /api/tags response is not valid JSON.", category="invalid_response")

    models_raw = data.get("models") or []
    available: list[str] = []
    for entry in models_raw:
        name = entry.get("name") or entry.get("model") or ""
        if name:
            available.append(name)

    print(f"  Ollama reachable at {_BASE_URL}")
    print(f"  Available models   ({len(available)} total):")
    for m in available:
        marker = " ← target" if _MODEL in m or m in _MODEL else ""
        print(f"    - {m}{marker}")

    return available


# ---------------------------------------------------------------------------
# Step 2: Verify target model is listed
# ---------------------------------------------------------------------------

def check_model_available(available: list[str]) -> None:
    """Confirm the target model appears in the tags list."""
    print(f"\n[2/3] Verifying model '{_MODEL}' is available ...")
    # Ollama may return names like "qwen3-embedding:0.6b" or with a digest suffix
    found = any(_MODEL in name or name in _MODEL for name in available)
    if not found:
        _fail(
            f"Model '{_MODEL}' not found in Ollama. "
            f"Pull it with: ollama pull {_MODEL}\n"
            f"  Available: {available}",
            category="model_not_found",
        )
    print(f"  Model '{_MODEL}' is listed.")


# ---------------------------------------------------------------------------
# Step 3: Embed probe text and validate response
# ---------------------------------------------------------------------------

def check_embed_dimensions() -> dict:
    """POST /api/embed with probe text and validate the returned vector."""
    print(f"\n[3/3] Running embed probe (model={_MODEL}) ...")
    payload = {"model": _MODEL, "input": [_PROBE_TEXT]}

    started = time.perf_counter()
    try:
        resp = httpx.post(_EMBED_URL, json=payload, timeout=_TIMEOUT)
    except httpx.ConnectError as exc:
        _fail(
            f"Cannot connect to Ollama at {_EMBED_URL}: {exc}",
            category="connection_error",
        )
    except httpx.TimeoutException:
        _fail(
            f"Embed request timed out after {_TIMEOUT}s (url={_EMBED_URL}, model={_MODEL}).",
            category="timeout",
        )
    except httpx.RequestError as exc:
        _fail(f"HTTP request error: {exc}", category="request_error")

    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    if not resp.is_success:
        _fail(
            f"Ollama /api/embed returned HTTP {resp.status_code} "
            f"(url={_EMBED_URL}, model={_MODEL}).",
            category="http_error",
        )

    try:
        data = resp.json()
    except Exception:
        _fail("Ollama /api/embed response is not valid JSON.", category="invalid_response")

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        _fail(
            f"Response missing 'embeddings' list (url={_EMBED_URL}, model={_MODEL}).\n"
            f"  Response keys: {list(data.keys())}",
            category="invalid_response",
        )

    if len(embeddings) != 1:
        _fail(
            f"Expected 1 embedding, got {len(embeddings)} (url={_EMBED_URL}, model={_MODEL}).",
            category="count_mismatch",
        )

    vector = embeddings[0]
    if not isinstance(vector, list) or not vector:
        _fail("Embedding[0] is empty or not a list.", category="invalid_response")

    if not all(_is_finite(v) for v in vector):
        _fail("Embedding[0] contains non-finite values.", category="non_numeric_values")

    dims = len(vector)

    if _EXPECTED_DIMS > 0 and dims != _EXPECTED_DIMS:
        _fail(
            f"Dimension mismatch: got {dims} but EMBEDDING_DIMENSIONS={_EXPECTED_DIMS}. "
            "Update EMBEDDING_DIMENSIONS in your .env to match the model output.",
            category="dimension_mismatch",
        )

    return {
        "model": _MODEL,
        "base_url": _BASE_URL,
        "dimensions": dims,
        "latency_ms": latency_ms,
        "vector_sample": [round(v, 6) for v in vector[:5]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Qwen3 Embedding Smoke Probe")
    print("=" * 60)
    _print_result("base_url:", _BASE_URL)
    _print_result("model:", _MODEL)
    _print_result("timeout_seconds:", _TIMEOUT)
    if _EXPECTED_DIMS > 0:
        _print_result("expected_dimensions:", _EXPECTED_DIMS)

    available = check_ollama_tags()
    check_model_available(available)
    result = check_embed_dimensions()

    print("\n" + "=" * 60)
    print("RESULT: OK")
    print("=" * 60)
    _print_result("model:", result["model"])
    _print_result("base_url:", result["base_url"])
    _print_result("dimensions:", result["dimensions"])
    _print_result("latency_ms:", result["latency_ms"])
    _print_result("vector_sample:", result["vector_sample"])
    print()
    print("Qwen3 embedding is ready.  To activate in the backend:")
    print(f"  EMBEDDING_PROVIDER=ollama")
    print(f"  EMBEDDING_MODEL={result['model']}")
    print(f"  EMBEDDING_DIMENSIONS={result['dimensions']}")
    print(f"  EMBEDDING_BASE_URL={result['base_url']}")
    print(f"  VECTOR_STORE_COLLECTION=knowledge_qwen3_embedding_06b")
    print()
    print("QWEN_SMOKE_PROBE_OK")


if __name__ == "__main__":
    main()
