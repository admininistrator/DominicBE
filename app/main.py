from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.api.endpoints import chat
from app.core.database import engine, Base, SQLALCHEMY_DATABASE_URL
import os

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: try to create tables, but don't crash if DB is unreachable yet
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as exc:
        logger.error("Could not connect to DB on startup: %s", exc)
        logger.error("DB URL (masked): %s", _mask_db_url(SQLALCHEMY_DATABASE_URL))
    yield


def _mask_db_url(url: str) -> str:
    """Hide password in DB URL for safe logging."""
    try:
        at = url.index("@")
        colon = url.index("://") + 3
        second_colon = url.index(":", colon)
        return url[:colon] + url[colon:second_colon] + ":***" + url[at:]
    except (ValueError, IndexError):
        return "(could not parse)"


app = FastAPI(title="Dominic Backend", lifespan=lifespan)


def _parse_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
    # fallback local (common Vite ports)
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.azurestaticapps\.net$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/env")
def debug_env():
    """Temporary endpoint to verify env-var visibility on Azure.  Remove after debugging."""
    return {
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "(not set)"),
        "DB_HOST": os.getenv("DB_HOST", "(not set)"),
        "DB_PORT": os.getenv("DB_PORT", "(not set)"),
        "DB_NAME": os.getenv("DB_NAME", "(not set)"),
        "DB_USER": os.getenv("DB_USER", "(not set)"),
        "DB_PASSWORD_SET": bool(os.getenv("DB_PASSWORD")),
        "ANTHROPIC_API_KEY_SET": bool(os.getenv("ANTHROPIC_API_KEY")),
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "(not set)"),
        "PORT": os.getenv("PORT", "(not set)"),
        "WEBSITE_HOSTNAME": os.getenv("WEBSITE_HOSTNAME", "(not set)"),
        "db_url_masked": _mask_db_url(SQLALCHEMY_DATABASE_URL),
        "allowed_origins": _parse_origins(),
    }


app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
