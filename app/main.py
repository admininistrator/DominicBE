from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
import logging
import sys

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.deps import get_current_user_optional
from app.crud import crud_auth
from app.api.endpoints import auth, chat, knowledge
from app.core.config import settings
from app.core.database import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("uvicorn.error")


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


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running"}


@app.get("/health")
def health():
    return {"ok": True}


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
