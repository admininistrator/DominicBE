"""API embedding provider probe (P02-T02).

Verifies that an API embedding provider is reachable, that the configured model
is available (where the API supports listing), and that a minimal embed call
returns a valid vector with the expected dimensions.

Usage
-----
Run directly from the DominicBE project root::

    python scripts/embedding_provider_probe.py

Or with environment variable overrides without a .env file::

    EMBEDDING_PROVIDER=api            \\
    EMBEDDING_MODEL=text-embedding-3-small \\
    EMBEDDING_API_TYPE=openai        \\
    EMBEDDING_API_KEY=sk-...         \\
    EMBEDDING_BASE_URL=https://api.openai.com/v1 \\
    python scripts/embedding_provider_probe.py

Exit codes
----------
0  All checks passed.
1  Any check failed.

Design constraints
------------------
- Does NOT require ingestion, a running FastAPI server, or a database.
- Does NOT require vector store, CRUD, endpoints, chat, or LlamaIndex.
- Does NOT log raw probe text or API keys.
- Reports: provider, model, api_type, dimensions (detected), latency_ms,
  base_url (host only — no path or key).
- Uses httpx for HTTP calls; uses GenericAPIProvider for the embed probe.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

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
# Configuration — read from settings or environment with safe fallback
# ---------------------------------------------------------------------------
try:
    from app.core.config import settings as _settings

    CONFIG_SOURCE = "settings"
    _PROVIDER: str = (_settings.embedding_provider or "api").strip().lower()
    _MODEL: str = _settings.embedding_model or ""
    _BASE_URL: str = (_settings.embedding_base_url or "").rstrip("/")
    _TIMEOUT: float = _settings.embedding_timeout_seconds or 60.0
    _EXPECTED_DIMS: int = _settings.embedding_dimensions or 0
    _API_KEY: str = _settings.embedding_api_key or ""
    _API_TYPE: str = (_settings.embedding_api_type or "").strip().lower()
    _API_VERSION: str = _settings.embedding_api_version or ""
    _API_HEADERS_RAW: str = _settings.embedding_api_headers or ""
    _COLLECTION: str = _settings.vector_store_collection or ""
except Exception:
    CONFIG_SOURCE = "environ"
    _PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "api").strip().lower()
    _MODEL = os.environ.get("EMBEDDING_MODEL", "")
    _BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
    _TIMEOUT = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60"))
    _EXPECTED_DIMS = int(os.environ.get("EMBEDDING_DIMENSIONS", "0"))
    _API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
    _API_TYPE = os.environ.get("EMBEDDING_API_TYPE", "").strip().lower()
    _API_VERSION = os.environ.get("EMBEDDING_API_VERSION", "")
    _API_HEADERS_RAW = os.environ.get("EMBEDDING_API_HEADERS", "")
    _COLLECTION = os.environ.get("VECTOR_STORE_COLLECTION", "")

# Parse custom headers JSON (if any)
_API_HEADERS: dict[str, str] = {}
if _API_HEADERS_RAW.strip():
    try:
        _API_HEADERS = json.loads(_API_HEADERS_RAW)
        if not isinstance(_API_HEADERS, dict):
            print(
                "WARNING: EMBEDDING_API_HEADERS is not a JSON object; ignoring.",
                file=sys.stderr,
            )
            _API_HEADERS = {}
    except (json.JSONDecodeError, TypeError) as exc:
        print(
            f"WARNING: EMBEDDING_API_HEADERS is malformed JSON: {exc}; ignoring.",
            file=sys.stderr,
        )
        _API_HEADERS = {}

# Normalise base_url: ensure we have at least scheme + host for connectivity checks
_PROBE_BASE_URL = _BASE_URL if _BASE_URL else "http://localhost"

# Probe text — short, generic, no sensitive content
_PROBE_TEXT = "embedding provider probe"

# API types that support model listing endpoints
_API_TYPES_WITH_MODEL_LISTING = frozenset({"openai", "ollama"})


# ---------------------------------------------------------------------------
# Helpers (local — avoids importing from app.services.embeddings.security to
# prevent triggering __init__.py -> factory.py -> config:Settings() chain)
# ---------------------------------------------------------------------------


def _get_host(url: str) -> str:
    """Extract hostname from a URL, stripping path and credentials."""
    parsed = urlparse(url)
    return parsed.netloc or parsed.hostname or url


def _mask_api_key(key: str) -> str:
    """Mask an API key for safe display (mirrors security.mask_api_key)."""
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return key[:3] + "..." + key[-4:]


def _sanitize_error_message(msg: str, api_key: str) -> str:
    """Strip API key from error message (mirrors security.sanitize_error_message)."""
    if not api_key or not msg:
        return msg
    if api_key in msg:
        msg = msg.replace(api_key, _mask_api_key(api_key))
    return msg


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
# Step 1: Connectivity / config validation
# ---------------------------------------------------------------------------


def _check_connectivity_ollama(client: httpx.Client) -> None:
    """Check Ollama reachability via /api/tags."""
    tags_url = f"{_PROBE_BASE_URL}/api/tags"
    try:
        resp = client.get(tags_url)
    except httpx.ConnectError as exc:
        _fail(
            f"Cannot connect to Ollama at {_get_host(_PROBE_BASE_URL)}. "
            "Check EMBEDDING_BASE_URL and that Ollama is running.\n"
            f"  Detail: {exc}",
            category="connection_error",
        )
    except httpx.TimeoutException:
        _fail(
            f"Connection to {tags_url} timed out after {_TIMEOUT}s.",
            category="timeout",
        )
    except httpx.RequestError as exc:
        _fail(f"HTTP request error: {exc}", category="request_error")

    if not resp.is_success:
        _fail(
            f"Ollama /api/tags returned HTTP {resp.status_code}.",
            category="http_error",
        )

    print(f"  Ollama reachable at {_get_host(_PROBE_BASE_URL)}")


def _check_connectivity_openai(client: httpx.Client) -> None:
    """Check OpenAI-compatible API reachability via /v1/models."""
    models_url = f"{_PROBE_BASE_URL.rstrip('/')}/v1/models"
    headers = {}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    if _API_HEADERS:
        headers.update(_API_HEADERS)
    try:
        resp = client.get(models_url, headers=headers)
    except httpx.ConnectError as exc:
        _fail(
            f"Cannot connect to {_get_host(_PROBE_BASE_URL)}. "
            "Check EMBEDDING_BASE_URL and that the API endpoint is correct.\n"
            f"  Detail: {exc}",
            category="connection_error",
        )
    except httpx.TimeoutException:
        _fail(
            f"Connection to {_get_host(_PROBE_BASE_URL)} timed out after {_TIMEOUT}s.",
            category="timeout",
        )
    except httpx.RequestError as exc:
        _fail(f"HTTP request error: {exc}", category="request_error")

    if not resp.is_success:
        _fail(
            f"API /v1/models returned HTTP {resp.status_code}. "
            "Check EMBEDDING_BASE_URL and EMBEDDING_API_KEY.",
            category="http_error",
        )

    print(f"  API reachable at {_get_host(_PROBE_BASE_URL)}")


def _check_connectivity_generic(client: httpx.Client) -> None:
    """Check generic API reachability with a lightweight GET."""
    try:
        resp = client.get(_PROBE_BASE_URL)
    except httpx.ConnectError as exc:
        _fail(
            f"Cannot connect to {_get_host(_PROBE_BASE_URL)}. "
            "Check EMBEDDING_BASE_URL.\n"
            f"  Detail: {exc}",
            category="connection_error",
        )
    except httpx.TimeoutException:
        _fail(
            f"Connection to {_get_host(_PROBE_BASE_URL)} timed out after {_TIMEOUT}s.",
            category="timeout",
        )
    except httpx.RequestError as exc:
        _fail(f"HTTP request error: {exc}", category="request_error")

    print(f"  Host reachable at {_get_host(_PROBE_BASE_URL)}")


def check_connectivity() -> None:
    """Step 1: Validate endpoint reachability."""
    print(f"\n[1/3] Checking connectivity to {_get_host(_PROBE_BASE_URL)} ...")

    effective_type = _API_TYPE or "openai"

    with httpx.Client(timeout=min(_TIMEOUT, 10.0)) as client:
        if effective_type == "ollama":
            _check_connectivity_ollama(client)
        elif effective_type == "openai":
            _check_connectivity_openai(client)
        else:
            _check_connectivity_generic(client)

    print("  Connectivity: OK")


# ---------------------------------------------------------------------------
# Step 2: Model availability check
# ---------------------------------------------------------------------------


def _check_model_ollama(client: httpx.Client) -> None:
    """Check that the configured model is available in Ollama."""
    tags_url = f"{_PROBE_BASE_URL}/api/tags"
    try:
        resp = client.get(tags_url)
    except httpx.RequestError as exc:
        _fail(f"Failed to fetch Ollama tags: {exc}", category="request_error")

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

    print(f"  Available models ({len(available)} total):")
    found = False
    for m in available:
        marker = " ← target" if _MODEL in m or m in _MODEL else ""
        if marker:
            found = True
        print(f"    - {m}{marker}")

    if not found and available:
        _fail(
            f"Model '{_MODEL}' not found in Ollama. "
            f"Pull it with: ollama pull {_MODEL}\n"
            f"  Available: {available}",
            category="model_not_found",
        )
    elif not found and not available:
        _fail(
            "Ollama reported no models. Pull a model with: ollama pull <model>",
            category="model_not_found",
        )

    print(f"  Model '{_MODEL}' is available.")


def _check_model_openai(client: httpx.Client) -> None:
    """Check that the configured model is available via OpenAI /v1/models."""
    models_url = f"{_PROBE_BASE_URL.rstrip('/')}/v1/models"
    headers = {}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    if _API_HEADERS:
        headers.update(_API_HEADERS)

    try:
        resp = client.get(models_url, headers=headers)
    except httpx.RequestError as exc:
        _fail(f"Failed to fetch model list: {exc}", category="request_error")

    if not resp.is_success:
        _fail(
            f"API /v1/models returned HTTP {resp.status_code}. "
            "Check EMBEDDING_BASE_URL and EMBEDDING_API_KEY.",
            category="http_error",
        )

    try:
        data = resp.json()
    except Exception:
        _fail("API /v1/models response is not valid JSON.", category="invalid_response")

    model_ids = [entry.get("id", "") for entry in data.get("data", []) if entry.get("id")]
    found = _MODEL in model_ids

    print(f"  Available models ({len(model_ids)} total):")
    for mid in sorted(model_ids):
        marker = " ← target" if mid == _MODEL else ""
        print(f"    - {mid}{marker}")

    if not found:
        _fail(
            f"Model '{_MODEL}' not found in API model list. "
            "Check EMBEDDING_MODEL for the correct model identifier.",
            category="model_not_found",
        )

    print(f"  Model '{_MODEL}' is available.")


def check_model_available() -> None:
    """Step 2: Verify the configured model is available (if API supports listing)."""
    effective_type = _API_TYPE or "openai"

    print(f"\n[2/3] Checking model availability for '{_MODEL}' (type={effective_type}) ...")

    if not _MODEL:
        print("  WARNING: No model configured (EMBEDDING_MODEL is empty). Skipping.")
        return

    if effective_type not in _API_TYPES_WITH_MODEL_LISTING:
        print(f"  Model listing not supported for api_type='{effective_type}'; skipping.")
        return

    with httpx.Client(timeout=min(_TIMEOUT, 10.0)) as client:
        if effective_type == "ollama":
            _check_model_ollama(client)
        elif effective_type == "openai":
            _check_model_openai(client)


# ---------------------------------------------------------------------------
# Step 3: Embed probe
# ---------------------------------------------------------------------------


def check_embed_probe() -> dict:
    """Step 3: Send a minimal embedding request and validate the response."""
    print(f"\n[3/3] Running embed probe with model='{_MODEL}' ...")

    # Lazy imports — only needed for this step.
    # Broad except catches settings/env issues from __init__.py chain.
    try:
        from app.services.embeddings.base import EmbeddingProviderError  # noqa: F811
        from app.services.embeddings.generic_api_provider import GenericAPIProvider
    except Exception as exc:
        _fail(
            f"Cannot import embedding provider modules: {exc}. "
            "Make sure the project is installed or the virtual environment is active.",
            category="configuration_error",
        )

    # Build custom headers dict
    custom_headers = dict(_API_HEADERS) if _API_HEADERS else None

    try:
        provider = GenericAPIProvider(
            model=_MODEL,
            base_url=_BASE_URL,
            api_key=_API_KEY,
            api_type=_API_TYPE,
            timeout_seconds=_TIMEOUT,
            batch_size=1,
            expected_dimensions=_EXPECTED_DIMS,
            api_version=_API_VERSION,
            custom_headers=custom_headers,
        )
    except EmbeddingProviderError as exc:
        _fail(
            _sanitize_error_message(str(exc), _API_KEY),
            category=exc.category,
        )
    except Exception as exc:
        _fail(
            _sanitize_error_message(f"Failed to initialise provider: {exc}", _API_KEY),
            category="configuration_error",
        )

    # Run the embed probe
    started = time.perf_counter()
    try:
        result = provider.embed_texts([_PROBE_TEXT])
    except EmbeddingProviderError as exc:
        _fail(
            _sanitize_error_message(str(exc), _API_KEY),
            category=exc.category,
        )
    except httpx.ConnectError as exc:
        _fail(
            _sanitize_error_message(
                f"Cannot connect to {_get_host(_BASE_URL)} during embed: {exc}",
                _API_KEY,
            ),
            category="connection_error",
        )
    except httpx.TimeoutException:
        _fail(
            f"Embed request timed out after {_TIMEOUT}s (model={_MODEL}).",
            category="timeout",
        )
    except httpx.RequestError as exc:
        _fail(
            _sanitize_error_message(f"HTTP request error during embed: {exc}", _API_KEY),
            category="request_error",
        )
    except Exception as exc:
        _fail(
            _sanitize_error_message(f"Unexpected error during embed: {exc}", _API_KEY),
            category="unknown",
        )

    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    # Validate response
    vectors = result.vectors
    if not isinstance(vectors, list) or not vectors:
        _fail(
            "Embedding response has no vectors.",
            category="invalid_response",
        )

    if len(vectors) != 1:
        _fail(
            f"Expected 1 embedding vector, got {len(vectors)}.",
            category="count_mismatch",
        )

    vector = vectors[0]
    if not isinstance(vector, list) or not vector:
        _fail("Embedding vector is empty or not a list.", category="invalid_response")

    if not all(_is_finite(v) for v in vector):
        _fail("Embedding vector contains non-finite values.", category="non_numeric_values")

    dims = len(vector)

    if _EXPECTED_DIMS > 0 and dims != _EXPECTED_DIMS:
        _fail(
            f"Dimension mismatch: got {dims} but EMBEDDING_DIMENSIONS={_EXPECTED_DIMS}. "
            "Update EMBEDDING_DIMENSIONS in your environment to match the model output.",
            category="dimension_mismatch",
        )

    return {
        "model": _MODEL,
        "base_url": _BASE_URL,
        "api_type": _API_TYPE or "openai",
        "dimensions": dims,
        "latency_ms": latency_ms,
        "vector_sample": [round(v, 6) for v in vector[:5]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Embedding Provider Probe")
    print("=" * 60)
    _print_result("config_source:", CONFIG_SOURCE)
    _print_result("provider:", _PROVIDER)
    _print_result("model:", _MODEL or "(not set)")
    _print_result("api_type:", _API_TYPE or "(not set — default: openai)")
    _print_result("base_url:", _get_host(_BASE_URL) if _BASE_URL else "(not set)")

    # Show API key status (masked) — never the raw key
    masked = _mask_api_key(_API_KEY)
    key_status = f"{masked} (set)" if _API_KEY else "(not set)"
    _print_result("api_key:", key_status)

    if _EXPECTED_DIMS > 0:
        _print_result("expected_dims:", _EXPECTED_DIMS)
    _print_result("timeout_seconds:", _TIMEOUT)
    if _COLLECTION:
        _print_result("collection:", _COLLECTION)

    # Step 1
    check_connectivity()

    # Step 2
    check_model_available()

    # Step 3
    result = check_embed_probe()

    # Success output
    print("\n" + "=" * 60)
    print("RESULT: OK")
    print("=" * 60)
    _print_result("provider:", _PROVIDER)
    _print_result("model:", result["model"])
    _print_result("api_type:", result["api_type"])
    _print_result("base_url:", _get_host(result["base_url"]))
    _print_result("dimensions:", result["dimensions"])
    _print_result("latency_ms:", result["latency_ms"])
    _print_result("vector_sample:", result["vector_sample"])

    # Ready-to-paste config block — use suggest_collection_name if available
    try:
        from app.services.embeddings.collection_naming import suggest_collection_name

        suggested_collection = suggest_collection_name(_PROVIDER, _MODEL)
    except Exception:
        suggested_collection = (
            f"knowledge_{_PROVIDER}_{_MODEL.replace('-', '_').replace(':', '_').replace('/', '_')}"
        )

    print()
    print("Embedding provider is ready. To use this provider, ensure your .env contains:")
    print(f"  EMBEDDING_PROVIDER={_PROVIDER}")
    print(f"  EMBEDDING_MODEL={result['model']}")
    print(f"  EMBEDDING_DIMENSIONS={result['dimensions']}")
    print(f"  EMBEDDING_BASE_URL={result['base_url']}")
    print(f"  EMBEDDING_API_TYPE={result['api_type']}")
    print(f"  VECTOR_STORE_COLLECTION={suggested_collection}")

    # Show key reminder (masked)
    if _API_KEY:
        print(f"  EMBEDDING_API_KEY={masked}  (already set — replace with actual key in .env)")

    # Collection naming tip
    print()
    print(f"TIP: Suggested collection name: {suggested_collection}")
    if _COLLECTION and _COLLECTION != suggested_collection:
        print(f"     Current VECTOR_STORE_COLLECTION={_COLLECTION}")
        print(f"     You may want to update to: {suggested_collection}")

    print()
    print("EMBEDDING_PROVIDER_PROBE_OK")


if __name__ == "__main__":
    main()
