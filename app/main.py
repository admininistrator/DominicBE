from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
import logging
import sys
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
from app.core.database import check_database_health, engine
from app.services.object_storage import check_object_storage_health
from app.services.vector_store import check_vector_store_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("uvicorn.error")

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
    logger.info("anthropic package version = %s", _get_package_version("anthropic"))
    logger.info("ANTHROPIC_API_KEY set = %s", bool(settings.anthropic_api_key))
    logger.info("ANTHROPIC_MODEL = %s", settings.anthropic_model)
    logger.info("ANTHROPIC_BASE_URL = %s", settings.anthropic_base_url or "(default)")
    logger.info("ANTHROPIC_FORCE_IPV4 = %s", settings.anthropic_force_ipv4)
    if settings.auth_secret_key == "change-this-in-production":
        logger.warning(
            "AUTH_SECRET_KEY is using the default value. Set a strong secret in non-local environments."
        )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connectivity check passed.")
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)

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
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.azurestaticapps\.net$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
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
        logger.info(
            "request completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            route_path,
            status_code,
            duration * 1000,
        )


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running"}


@app.get("/health")
def health():
    checks = {
        "postgres": check_database_health(),
        "minio": check_object_storage_health(),
        "qdrant": check_vector_store_health(),
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


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/env", include_in_schema=False)
def debug_env(current_user=Depends(get_current_user_optional)):
    if not settings.enable_debug_env:
        raise HTTPException(status_code=404, detail="Not found")

    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    if not crud_auth.is_effective_admin_username(getattr(current_user, "username", None)):
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
        "anthropic_api_key_set": bool(settings.anthropic_api_key),
        "anthropic_model": settings.anthropic_model,
        "anthropic_base_url": settings.anthropic_base_url or "(default)",
        "anthropic_force_ipv4": settings.anthropic_force_ipv4,
        "db_url_masked": _mask_db_url(settings.sqlalchemy_database_url),
    }


app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
