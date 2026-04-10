from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.api.endpoints import chat
from app.core.database import engine, Base
import os

# NOTE: Tam giu create_all cho nhanh. Production chuan nen doi sang migration.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Dominic Backend")

def _parse_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
    # fallback local (common Vite ports)
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
