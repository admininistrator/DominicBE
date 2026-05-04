from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from types import SimpleNamespace

os.environ["DEBUG"] = "false"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import deps
from app.api.endpoints import chat
from app.core.database import Base
from app.core.rate_limit import FixedWindowRateLimiter, RateLimitRule
from app.main import app
from app.models.chat_models import ChatSession, User
import app.main as main_module


GENERIC_INTERNAL_ERROR = "An internal server error occurred."


@pytest.fixture()
def app_instance(monkeypatch):
    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", noop_lifespan)
    app.dependency_overrides.clear()
    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
def sqlite_session_factory():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    try:
        yield testing_session_local
    finally:
        engine.dispose()
        os.remove(db_path)


def _override_current_user(username: str = "tester", role: str = "user"):
    return lambda: SimpleNamespace(username=username, role=role)


def _override_get_db(session_factory):
    def _override():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    return _override


def test_json_internal_errors_are_sanitized(app_instance, monkeypatch):
    secret = "super-secret-db-url"

    def broken_usage(*args, **kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(chat, "get_usage", broken_usage)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/chat/usage/me")

    assert response.status_code == 500
    assert response.json()["detail"] == GENERIC_INTERNAL_ERROR
    assert secret not in response.text


def test_stream_internal_errors_are_sanitized(app_instance, monkeypatch):
    secret = "super-secret-db-url"

    def broken_stream(*args, **kwargs):
        raise RuntimeError(secret)
        yield

    monkeypatch.setattr(chat, "handle_chat_stream", broken_stream)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    payload = {
        "username": "tester",
        "session_id": 1,
        "message": "hello",
        "knowledge_document_id": None,
        "use_web_search": False,
        "images": [],
        "image_media_types": [],
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        with client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            stream_text = "".join(response.iter_text())

    assert response.status_code == 200
    assert GENERIC_INTERNAL_ERROR in stream_text
    assert secret not in stream_text


def test_manual_session_rename_endpoint_works(app_instance, sqlite_session_factory):
    seed_db = sqlite_session_factory()
    try:
        seed_db.add(User(username="tester", password_hash="hash", role="user"))
        seed_db.commit()
        session = ChatSession(username="tester", title="Chat 1")
        seed_db.add(session)
        seed_db.commit()
        seed_db.refresh(session)
        session_id = session.id
    finally:
        seed_db.close()

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json={"title": "  Sprint Plan  "},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Sprint Plan"
    assert payload["username"] == "tester"


def test_chat_rate_limits_sync_and_stream_endpoints(app_instance, monkeypatch):
    def fake_handle_chat(*args, **kwargs):
        return {
            "reply": "ok",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "request_id": "req-1",
            "sources": [],
            "retrieval": None,
        }

    def fake_handle_chat_stream(*args, **kwargs):
        yield {"event": "message", "data": {"delta": "ok"}}
        yield {"event": "done", "data": {"reply": "ok"}}

    monkeypatch.setattr(chat, "handle_chat", fake_handle_chat)
    monkeypatch.setattr(chat, "handle_chat_stream", fake_handle_chat_stream)
    monkeypatch.setattr(
        main_module,
        "RATE_LIMIT_RULES",
        [
            RateLimitRule("chat.send", frozenset({"POST"}), "/api/chat/", 1, 60),
            RateLimitRule("chat.stream", frozenset({"POST"}), "/api/chat/stream", 1, 60),
        ],
    )
    monkeypatch.setattr(main_module, "RATE_LIMITER", FixedWindowRateLimiter())
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", True)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    payload = {
        "username": "tester",
        "session_id": 1,
        "message": "hello",
        "knowledge_document_id": None,
        "use_web_search": False,
        "images": [],
        "image_media_types": [],
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        first_chat = client.post("/api/v1/chat/", json=payload)
        second_chat = client.post("/api/v1/chat/", json=payload)
        with client.stream("POST", "/api/v1/chat/stream", json=payload) as first_stream_response:
            first_stream_text = "".join(first_stream_response.iter_text())
        second_stream = client.post("/api/v1/chat/stream", json=payload)

    assert first_chat.status_code == 200
    assert second_chat.status_code == 429
    assert second_chat.json()["scope"] == "chat.send"
    assert first_stream_response.status_code == 200
    assert "event: done" in first_stream_text
    assert second_stream.status_code == 429
    assert second_stream.json()["scope"] == "chat.stream"


def test_cors_allows_explicit_origin_and_denies_unlisted_azure_origin(app_instance):
    allowed_headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    denied_headers = {
        "Origin": "https://evil-preview.azurestaticapps.net",
        "Access-Control-Request-Method": "POST",
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        allowed = client.options("/api/v1/chat/", headers=allowed_headers)
        denied = client.options("/api/v1/chat/", headers=denied_headers)

    assert allowed.status_code == 200
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert denied.status_code == 400
    assert denied.headers.get("access-control-allow-origin") is None


def test_redundant_username_scoped_chat_routes_do_not_exist(app_instance):
    with TestClient(app_instance, raise_server_exceptions=False) as client:
        usage_response = client.get("/api/v1/chat/usage/testuser")
        sessions_response = client.get("/api/v1/chat/sessions/testuser")
        messages_response = client.get("/api/v1/chat/sessions/testuser/123/messages")

    assert usage_response.status_code == 401
    assert sessions_response.status_code == 405
    assert messages_response.status_code == 404


def test_auth_refresh_and_logout_revoke_existing_tokens(app_instance, sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "tester",
                "password": "StrongPass1",
                "confirm_password": "StrongPass1",
            },
        )

        assert register_response.status_code == 201
        register_payload = register_response.json()
        assert register_payload["refresh_token"]

        me_before_logout = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {register_payload['access_token']}"},
        )
        refresh_response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": register_payload["refresh_token"]},
        )
        logout_response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {register_payload['access_token']}"},
        )
        me_after_logout = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {register_payload['access_token']}"},
        )
        refresh_after_logout = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": register_payload["refresh_token"]},
        )

    assert me_before_logout.status_code == 200
    assert refresh_response.status_code == 200
    assert refresh_response.json()["access_token"] != register_payload["access_token"]
    assert refresh_response.json()["refresh_token"] != register_payload["refresh_token"]
    assert logout_response.status_code == 200
    assert logout_response.json()["message"] == "Logged out successfully."
    assert me_after_logout.status_code == 401
    assert me_after_logout.json()["detail"] == "Token has been revoked."
    assert refresh_after_logout.status_code == 401
    assert refresh_after_logout.json()["detail"] == "Refresh token has been revoked."


def test_change_password_revokes_existing_access_tokens(app_instance, sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "tester2",
                "password": "StrongPass1",
                "confirm_password": "StrongPass1",
            },
        )

        access_token = register_response.json()["access_token"]
        change_response = client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "old_password": "StrongPass1",
                "new_password": "NewStrongPass2",
                "confirm_new_password": "NewStrongPass2",
            },
        )
        me_after_change = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        login_with_new_password = client.post(
            "/api/v1/auth/login",
            json={"username": "tester2", "password": "NewStrongPass2"},
        )

    assert change_response.status_code == 200
    assert me_after_change.status_code == 401
    assert me_after_change.json()["detail"] == "Token has been revoked."
    assert login_with_new_password.status_code == 200
    assert login_with_new_password.json()["refresh_token"]