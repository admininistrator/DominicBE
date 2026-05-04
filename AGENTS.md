# Project Guidelines

## Scope
- Primary backend code lives under `app/`; top-level markdown files document the current verified state and deployment paths.
- Use `README.md` for the current feature/status baseline and `DEPLOY_AWS_EC2_DOCKER.md` for deployment-specific detail instead of copying them here.

## Architecture
- `app/main.py` is the FastAPI entrypoint. It mounts `auth`, `chat`, and `knowledge` routers under `/api/v1/*` and keeps `/api/*` legacy aliases for compatibility.
- Keep HTTP-layer validation and response shaping in `app/api/endpoints`, data access in `app/crud`, schema contracts in `app/schemas`, and business logic in `app/services`.
- `app/core/config.py` is the source of truth for environment-driven behavior. Add new settings there rather than reading env vars ad hoc.
- Knowledge and RAG changes usually span `app/services/knowledge_service.py`, `app/services/retrieval_service.py`, `app/services/object_storage.py`, and `app/services/vector_store.py`.
- Any schema change should ship with a matching Alembic revision in `alembic/versions`.

## Build and Test
- Use only the repo Python environment: `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux.
- Preferred local run: `powershell -ExecutionPolicy Bypass -File scripts/dev_backend.ps1`
- Migration: `.venv\Scripts\python.exe -m alembic upgrade head`
- Narrow regression test: `.venv\Scripts\python.exe -m pytest tests/test_api_regressions.py`
- Smoke scripts for touched slices live in `scripts/`, especially `auth_smoke_test.py`, `knowledge_smoke_test.py`, `rag_chat_smoke_test.py`, `rag_eval_smoke_test.py`, `test_image_processor.py`, and `test_ocr_injection.py`.

## Conventions
- Prefer `/api/v1` when adding or updating clients, docs, or tests. Keep the legacy `/api` aliases unless the change explicitly removes backward compatibility.
- Preserve request-id propagation, metrics, rate limiting, and sanitized error handling when editing `app/main.py` or endpoint error paths.
- Keep storage backends configurable through settings such as `DATABASE_URL`, `OBJECT_STORAGE_PROVIDER`, and `VECTOR_STORE_PROVIDER`; avoid local-only assumptions in service logic.
- `tests/test_api_regressions.py` uses `TestClient`, SQLite, and monkeypatch-based seams for fast API boundary checks. Extend that style first for endpoint regressions before reaching for heavier integration tests.
- When auth, chat, knowledge, storage, or deployment behavior changes, update the matching docs and smoke/regression tests in the same change.