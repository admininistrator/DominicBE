from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import sys
from threading import Thread
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.api.deps import get_current_user_optional
from app.crud import crud_auth
from app.api.endpoints import auth, chat, knowledge
from app.core.config import settings
from app.core.database import SessionLocal, check_database_health, engine
from app.core.logging import (
    clear_log_context,
    configure_logging,
    get_log_context,
    get_logger,
    restore_log_context,
    set_log_context,
)
from app.core.migrations import MigrationValidationError, validate_database_migrations
from app.core.rate_limit import (
    attach_rate_limit_headers,
    build_rate_limiter,
    build_default_rate_limit_rules,
    build_rate_limited_response,
    find_matching_rule,
    get_client_ip,
)
from app.services.knowledge_service import recover_pending_ingestion_jobs
from app.services.object_storage import check_object_storage_health
from app.services.vector_store import check_vector_store_health


def check_embedding_health() -> dict:
    """Return embedding provider readiness without blocking local defaults.

    - local provider: always OK (no external service required).
    - ollama provider: performs a lightweight GET /api/tags probe.
      Reports unavailable when Ollama is unreachable; never leaks document text.
    """
    provider = (settings.embedding_provider or "local").strip().lower()
    model = settings.embedding_model or "local-hash-v1"
    base_url = (settings.embedding_base_url or "http://localhost:11434").rstrip("/")

    if provider == "local":
        return {
            "ok": True,
            "provider": provider,
            "model": model,
            "detail": "local provider requires no external service",
        }

    if provider == "ollama":
        tags_url = f"{base_url}/api/tags"
        try:
            import httpx

            timeout = min(float(settings.embedding_timeout_seconds or 60.0), 10.0)
            resp = httpx.get(tags_url, timeout=timeout)
            if resp.is_success:
                # Check whether the configured model is listed
                data = resp.json()
                models_raw = data.get("models") or []
                available = [
                    (entry.get("name") or entry.get("model") or "")
                    for entry in models_raw
                ]
                model_found = any(model in name or name in model for name in available)
                return {
                    "ok": True,
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "model_listed": model_found,
                    "detail": "ollama reachable" + ("" if model_found else f"; model '{model}' not yet pulled"),
                }
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "detail": f"ollama /api/tags returned HTTP {resp.status_code}",
            }
        except ImportError:
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "detail": "httpx not installed; cannot probe ollama",
            }
        except httpx.ConnectError as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "detail": f"ollama connection refused: {type(exc).__name__}",
            }
        except httpx.TimeoutException as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "detail": f"ollama timeout: {type(exc).__name__}",
            }
        except httpx.HTTPStatusError as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "detail": f"ollama HTTP error: {type(exc).__name__}",
            }
        except httpx.RequestError as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "detail": f"ollama request failed: {type(exc).__name__}",
            }

    if provider == "api":
        # Lazy imports to avoid overhead for local/ollama paths
        from app.services.embeddings.generic_api_provider import GenericAPIProvider

        api_type = (settings.embedding_api_type or "").strip().lower()
        api_key = settings.embedding_api_key or ""
        timeout = min(float(settings.embedding_timeout_seconds or 60.0), 10.0)

        # Parse custom headers from JSON string setting
        custom_headers: dict = {}
        raw_headers = (settings.embedding_api_headers or "").strip()
        if raw_headers:
            try:
                custom_headers = json.loads(raw_headers)
            except (json.JSONDecodeError, TypeError):
                pass

        try:
            provider_instance = GenericAPIProvider(
                model=model,
                base_url=base_url,
                api_key=api_key,
                api_type=api_type,
                timeout_seconds=timeout,
                batch_size=1,
                expected_dimensions=settings.embedding_dimensions or 0,
                api_version=settings.embedding_api_version or "",
                custom_headers=custom_headers,
            )

            started = perf_counter()
            result = provider_instance.embed_texts(["health check probe"])
            elapsed_ms = round((perf_counter() - started) * 1000, 2)

            dims = len(result.vectors[0]) if result.vectors else 0

            # NOTE: api_key intentionally excluded from response (even masked)
            return {
                "ok": True,
                "provider": provider,
                "model": model,
                "api_type": api_type,
                "dimensions": dims,
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            # Sanitize error to strip any accidental API key leakage
            detail = str(exc)
            if api_key:
                from app.services.embeddings.security import sanitize_error_message

                detail = sanitize_error_message(detail, api_key)

            category = getattr(exc, "category", "")
            msg = (
                f"api provider error: {category}"
                if category
                else f"api provider error: {type(exc).__name__}"
            )
            return {
                "ok": False,
                "provider": provider,
                "model": model,
                "api_type": api_type,
                "detail": msg,
            }

    return {
        "ok": False,
        "provider": provider,
        "model": model,
        "detail": f"unknown EMBEDDING_PROVIDER={provider!r}",
    }

configure_logging(logging.INFO)
logger = get_logger(__name__)
API_LEGACY_PREFIX = "/api"
API_V1_PREFIX = "/api/v1"

HTTP_REQUESTS_TOTAL = Counter(
    "dominic_http_requests_total",
    "Total HTTP requests handled by the backend.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "dominic_http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "dominic_http_requests_in_progress",
    "Current in-flight HTTP requests.",
)
RATE_LIMIT_RULES = build_default_rate_limit_rules(settings)
RATE_LIMIT_ENTRY_TTL_SECONDS = max(
    [rule.window_seconds for rule in RATE_LIMIT_RULES if rule.limit > 0],
    default=settings.rate_limit_cleanup_interval_seconds,
)
RATE_LIMITER = build_rate_limiter(settings, session_factory=SessionLocal)


def _get_package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "(not installed)"


def _mask_db_url(url: str) -> str:
    try:
        at = url.index("@")
        colon = url.index("://") + 3
        second_colon = url.index(":", colon)
        return url[:colon] + url[colon:second_colon] + ":***" + url[at:]
    except (ValueError, IndexError):
        return "(could not parse)"


def _ensure_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    return request_id


def _start_ingestion_recovery_worker() -> Thread | None:
    if not settings.ingestion_recovery_enabled or settings.ingestion_recovery_max_jobs <= 0:
        return None

    def _recover_jobs():
        try:
            summary = recover_pending_ingestion_jobs(
                SessionLocal,
                recovery_limit=settings.ingestion_recovery_max_jobs,
                stale_after_seconds=settings.ingestion_stuck_job_timeout_seconds,
            )
            if summary["selected_count"]:
                logger.info(
                    "ingestion recovery completed selected=%s queued=%s stale_processing=%s recovered=%s failed=%s",
                    summary["selected_count"],
                    summary["queued_count"],
                    summary["stale_processing_count"],
                    summary["recovered_count"],
                    summary["failed_count"],
                )
        except Exception:
            logger.exception("ingestion recovery failed at startup")

    worker = Thread(target=_recover_jobs, name="ingestion-recovery", daemon=True)
    worker.start()
    return worker


def _mount_api_router(*, router, segment: str, tags: list[str]):
    app.include_router(router, prefix=f"{API_V1_PREFIX}/{segment}", tags=tags)
    app.include_router(
        router,
        prefix=f"{API_LEGACY_PREFIX}/{segment}",
        tags=tags,
        include_in_schema=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== %s starting ===", settings.app_name)
    logger.info("ENVIRONMENT = %s", settings.environment)
    logger.info("PORT = %s", settings.port)
    logger.info("DB URL (masked) = %s", _mask_db_url(settings.sqlalchemy_database_url))
    logger.info("CORS_ORIGINS = %s", settings.cors_origins)
    logger.info(
        "AUTH_ACCESS_TOKEN_EXPIRE_MINUTES = %s",
        settings.auth_access_token_expire_minutes,
    )
    logger.info("litellm package version = %s", _get_package_version("litellm"))
    logger.info("GITHUB_COPILOT_API_KEY set = %s", bool(settings.github_copilot_api_key))
    logger.info("MODEL_GITHUB_COPILOT = %s", settings.github_copilot_model_name)
    logger.info("LLM_MODEL = %s", settings.llm_model or "(resolved from MODEL_GITHUB_COPILOT)")
    logger.info("NINEROUTER_BASE_URL = %s", settings.ninerouter_base_url)
    logger.info("LLM_CONTEXT_WINDOW = %s", settings.llm_context_window)
    if settings.auth_secret_key == "change-this-in-production":
        logger.warning(
            "AUTH_SECRET_KEY is using the default value. Set a strong secret in non-local environments."
        )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            if settings.migration_validation_enabled:
                migration_result = validate_database_migrations(
                    connection,
                    mode=settings.migration_validation_mode,
                )
                if migration_result.is_current:
                    logger.info(
                        "Database migration validation passed current_revision=%s",
                        ",".join(migration_result.current_revisions) or "(none)",
                    )
                else:
                    logger.warning(
                        "Database migration validation warning %s",
                        migration_result.describe(),
                    )
        logger.info("Database connectivity check passed.")
    except MigrationValidationError as exc:
        logger.error("Database migration validation failed: %s", exc)
        raise
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)

    _start_ingestion_recovery_worker()

    logger.info("=== %s ready ===", settings.app_name)
    yield
    logger.info("=== %s shutting down ===", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not settings.rate_limit_enabled:
        return await call_next(request)

    rule = find_matching_rule(
        RATE_LIMIT_RULES,
        method=request.method,
        path=request.url.path,
    )
    if rule is None:
        return await call_next(request)

    request_id = _ensure_request_id(request)
    client_ip = get_client_ip(request, trust_proxy_headers=settings.rate_limit_trust_proxy_headers)
    decision = RATE_LIMITER.check(rule=rule, key=client_ip)
    if not decision.allowed:
        previous_context = get_log_context()
        set_log_context(
            request_id=request_id,
            client_ip=client_ip,
            http_method=request.method,
            http_path=request.url.path,
            username=getattr(request.state, "auth_username", None),
        )
        logger.warning(
            "rate limit exceeded scope=%s",
            rule.name,
        )
        response = build_rate_limited_response(rule=rule, decision=decision)
        response.headers["X-Request-ID"] = request_id
        restore_log_context(previous_context)
        return response

    response = await call_next(request)
    return attach_rate_limit_headers(response, decision)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = _ensure_request_id(request)
    client_ip = get_client_ip(request, trust_proxy_headers=settings.rate_limit_trust_proxy_headers)
    set_log_context(
        request_id=request_id,
        client_ip=client_ip,
        http_method=request.method,
        http_path=request.url.path,
        username=getattr(request.state, "auth_username", None),
    )
    started_at = perf_counter()
    status_code = 500
    response: Response | None = None
    route_path = request.url.path

    HTTP_REQUESTS_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        status_code = response.status_code
        route = request.scope.get("route")
        route_path = getattr(route, "path", route_path)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        HTTP_REQUESTS_IN_PROGRESS.dec()
        route = request.scope.get("route")
        route_path = getattr(route, "path", route_path)
        duration = perf_counter() - started_at
        HTTP_REQUESTS_TOTAL.labels(request.method, route_path, str(status_code)).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(request.method, route_path).observe(duration)
        set_log_context(
            http_path=route_path,
            username=getattr(request.state, "auth_username", None),
        )
        logger.info(
            "request completed status=%s duration_ms=%.2f",
            status_code,
            duration * 1000,
        )
        clear_log_context()


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "api_versions": {
            "current": "v1",
            "supported": ["v1"],
            "default_base_path": API_V1_PREFIX,
            "legacy_base_path": API_LEGACY_PREFIX,
        },
    }


@app.get("/health")
def health():
    checks = {
        "postgres": check_database_health(),
        "minio": check_object_storage_health(),
        "qdrant": check_vector_store_health(),
        "embedding": check_embedding_health(),
    }
    payload = {
        "ok": all(check.get("ok") for check in checks.values()),
        "service": settings.app_name,
        "dependencies": checks,
    }
    return _health_response(payload)


def _health_response(payload: dict):
    return JSONResponse(status_code=200 if payload.get("ok") else 503, content=payload)


@app.get("/health/postgres")
def health_postgres():
    return _health_response(check_database_health())


@app.get("/health/minio")
def health_minio():
    return _health_response(check_object_storage_health())


@app.get("/health/qdrant")
def health_qdrant():
    return _health_response(check_vector_store_health())


@app.get("/health/embedding")
def health_embedding():
    return _health_response(check_embedding_health())


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/env", include_in_schema=False)
def debug_env(current_user=Depends(get_current_user_optional)):
    if not settings.enable_debug_env:
        raise HTTPException(status_code=404, detail="Not found")

    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required.")

    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "cors_origins": settings.cors_origins,
        "db_host": settings.db_host,
        "db_port": settings.db_port,
        "db_name": settings.db_name,
        "db_user": settings.db_user,
        "db_password_set": bool(settings.db_password),
        "github_copilot_api_key_set": bool(settings.github_copilot_api_key),
        "default_chat_model": settings.github_copilot_model_name,
        "llm_model": settings.llm_model or "(resolved from MODEL_GITHUB_COPILOT)",
        "supported_chat_models": settings.supported_chat_models,
        "ninerouter_base_url": settings.ninerouter_base_url,
        "llm_context_window": settings.llm_context_window,
        "db_url_masked": _mask_db_url(settings.sqlalchemy_database_url),
    }


_mount_api_router(router=auth.router, segment="auth", tags=["Auth"])
_mount_api_router(router=chat.router, segment="chat", tags=["Chat"])
_mount_api_router(router=knowledge.router, segment="knowledge", tags=["Knowledge"])
