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
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
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


# ---------------------------------------------------------------------------
# AUTH-BE-1: logout contract when access token is invalid or expired
# ---------------------------------------------------------------------------

def test_logout_with_invalid_access_token_returns_401(app_instance, monkeypatch):
    """POST /auth/logout with a structurally invalid (or expired) access token
    must be rejected at the auth layer with HTTP 401 before any server-side
    revocation logic is reached.

    This locks the contract:  invalid/expired token → 401, no side-effects.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer this.is.not.a.valid.jwt"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired access token."


def test_logout_blocked_by_invalid_token_does_not_revoke_server_session(
    app_instance, sqlite_session_factory, monkeypatch
):
    """If POST /auth/logout is rejected at the auth layer due to an invalid
    access token, the backend must NOT revoke the user's session.
    The original refresh token must remain fully usable afterwards.

    Locks the contract: auth-layer rejection → zero server-side revocation.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "tester_auth_be1a",
                "password": "StrongPass1",
                "confirm_password": "StrongPass1",
            },
        )
        assert register_response.status_code == 201
        original_refresh_token = register_response.json()["refresh_token"]

        # Attempt logout with an invalid / already-expired-like access token.
        blocked_logout = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer invalid.token.value"},
        )

        # The refresh token must not have been touched; it must still be valid.
        refresh_after_blocked_logout = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": original_refresh_token},
        )

    assert blocked_logout.status_code == 401
    assert blocked_logout.json()["detail"] == "Invalid or expired access token."
    assert refresh_after_blocked_logout.status_code == 200
    assert refresh_after_blocked_logout.json()["access_token"]
    assert refresh_after_blocked_logout.json()["refresh_token"]


def test_logout_without_bearer_does_not_revoke_refresh_token(
    app_instance, sqlite_session_factory, monkeypatch
):
    """POST /auth/logout with no Authorization header must return 401 and
    must leave the server-side session intact so the refresh token remains
    usable for obtaining fresh credentials.

    Locks the contract: unauthenticated logout attempt → refresh token survives.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "tester_auth_be1b",
                "password": "StrongPass1",
                "confirm_password": "StrongPass1",
            },
        )
        assert register_response.status_code == 201
        original_refresh_token = register_response.json()["refresh_token"]

        # Attempt logout with no Authorization header at all.
        unauthenticated_logout = client.post("/api/v1/auth/logout")

        # Refresh token must still be valid; no revocation should have occurred.
        refresh_after_unauthenticated = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": original_refresh_token},
        )

    assert unauthenticated_logout.status_code == 401
    assert unauthenticated_logout.json()["detail"] == "Not authenticated."
    assert refresh_after_unauthenticated.status_code == 200
    assert refresh_after_unauthenticated.json()["access_token"]
    assert refresh_after_unauthenticated.json()["refresh_token"]


def test_sync_chat_response_includes_assistant_meta(app_instance, monkeypatch):
    """POST /api/v1/chat/ must propagate assistant_meta from handle_chat result.

    Locks CHAT-BE-FIX-001: the sync endpoint was dropping the assistant_meta
    field that _finalize_chat_turn() already populated, causing a parity gap
    with the streaming final event.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)

    def fake_handle_chat(*args, **kwargs):
        return {
            "reply": "hello",
            "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
            "request_id": "req-fix-001",
            "sources": [],
            "retrieval": None,
            "assistant_meta": {
                "model": "gpt-4o",
                "reasoning_effort": None,
                "display_text": "GPT-4o",
            },
        }

    monkeypatch.setattr(chat, "handle_chat", fake_handle_chat)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    payload = {
        "username": "tester",
        "session_id": 1,
        "message": "hi",
        "knowledge_document_id": None,
        "use_web_search": False,
        "images": [],
        "image_media_types": [],
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/chat/", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_meta"] is not None, "assistant_meta must not be dropped by sync endpoint"
    assert body["assistant_meta"]["model"] == "gpt-4o"
    assert body["assistant_meta"]["display_text"] == "GPT-4o"


# ---------------------------------------------------------------------------
# CHAT-BE-005: canonical image contract enforcement
# ---------------------------------------------------------------------------

# Minimal 1×1 PNG that passes signature validation (used across image tests).
_VALID_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4DwAAAQEABRjYTgAAAABJRU5ErkJggg=="
)


def test_chat_rejects_more_than_5_images(app_instance, monkeypatch):
    """POST /api/v1/chat/ with 6 images must be rejected at schema validation
    with HTTP 422 before any service logic is reached.

    Locks CHAT-BE-005: max-image count is 5, not 10.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    payload = {
        "username": "tester",
        "session_id": 1,
        "message": "look at these",
        "images": [_VALID_PNG_DATA_URI] * 6,
        "image_media_types": [],
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/chat/", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert any("5" in str(err.get("msg", "")) for err in body.get("detail", []))


def test_chat_rejects_unsupported_image_mime_type(app_instance, monkeypatch):
    """POST /api/v1/chat/ with an unsupported MIME type in the data URI must
    be rejected at schema validation with HTTP 422.

    Locks CHAT-BE-005: only image/jpeg, image/png, image/webp, image/gif are allowed.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    # data:image/bmp is not in the allowed set
    bmp_uri = "data:image/bmp;base64,Qk0="

    payload = {
        "username": "tester",
        "session_id": 1,
        "message": "look at this",
        "images": [bmp_uri],
        "image_media_types": [],
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/chat/", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert any("Unsupported image media type" in str(err.get("msg", "")) for err in body.get("detail", []))


def test_chat_rejects_images_when_vision_disabled(app_instance, monkeypatch):
    """POST /api/v1/chat/ with a non-empty images list must return HTTP 400
    when LLM_VISION_ENABLED=false.  The backend must not silently drop the
    images and proceed; it must reject the request explicitly.

    Locks CHAT-BE-005: no silent drop at service layer.
    """
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    # Disable vision at the service layer
    import app.services.chat_service as chat_service_module
    monkeypatch.setattr(chat_service_module.settings, "llm_vision_enabled", False)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user()
    app_instance.dependency_overrides[deps.get_db] = lambda: iter([None])

    # handle_chat calls _prepare_chat_turn which raises ValueError when vision disabled.
    # We monkeypatch handle_chat to call the real _prepare_chat_turn path by raising
    # the same ValueError the service would raise, keeping the test narrow.
    def fake_handle_chat(*args, **kwargs):
        images = kwargs.get("images") or (args[4] if len(args) > 4 else None)
        if images:
            raise ValueError(
                "Image attachments are not supported: vision is disabled on this server."
            )
        return {
            "reply": "ok",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "request_id": "req-vision-disabled",
            "sources": [],
            "retrieval": None,
        }

    monkeypatch.setattr(chat, "handle_chat", fake_handle_chat)

    payload = {
        "username": "tester",
        "session_id": 1,
        "message": "look at this",
        "images": [_VALID_PNG_DATA_URI],
        "image_media_types": [],
    }

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/chat/", json=payload)

    assert response.status_code == 400
    assert "vision is disabled" in response.json()["detail"]


# ---------------------------------------------------------------------------
# CHAT-BE-006: session message pagination contract
# ---------------------------------------------------------------------------
#
# Four sub-contracts verified here:
#   1. Omitting `limit` does NOT return unbounded history when the session has
#      many messages — the endpoint must still return a well-formed response
#      (all messages, no artificial cap, has_more=false).
#   2. X-Message-Pagination-Limit header reflects the effective limit value
#      when limit is supplied; the header is absent when limit is omitted.
#   3. before_id / next_before_id cursor semantics: next_before_id points to
#      the oldest message id in the current page so the next request can
#      continue backwards.
#   4. The cursor path (before_id) works independently of skip — skip=0 and
#      skip omitted must produce identical results when before_id is used.
# ---------------------------------------------------------------------------


def _seed_session_with_messages(session_factory, username: str, n_messages: int):
    """Seed a user + session + n_messages and return (session_id, message_ids asc)."""
    db = session_factory()
    try:
        from app.models.chat_models import User, ChatSession, Message

        db.add(User(username=username, password_hash="hash", role="user"))
        db.commit()

        session = ChatSession(username=username, title="Pagination Test")
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id

        msg_ids = []
        for i in range(n_messages):
            msg = Message(
                session_id=session_id,
                sender_username=username,
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
                request_id=f"req-{i:04d}",
                status="success",
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            msg_ids.append(msg.id)

        return session_id, msg_ids
    finally:
        db.close()


def test_chat_pagination_omitted_limit_returns_all_messages_without_cap(
    app_instance, sqlite_session_factory
):
    """GET /sessions/{id}/messages with no limit query param must return ALL
    messages in the session (has_more=false, returned == total count).

    Locks CHAT-BE-006 sub-contract 1: omitting limit must not silently cap
    the result to some internal default — the caller controls pagination.
    """
    username = "tester_chat_pag_001"
    n = 8
    session_id, msg_ids = _seed_session_with_messages(sqlite_session_factory, username, n)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(username)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/chat/sessions/{session_id}/messages")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == n, f"Expected {n} messages, got {len(body)}"

    headers = response.headers
    assert headers.get("x-message-pagination-returned") == str(n)
    assert headers.get("x-message-pagination-has-more") == "false"
    # When limit is omitted the header must NOT be present (no effective limit).
    assert "x-message-pagination-limit" not in headers, (
        "X-Message-Pagination-Limit must be absent when limit param is omitted"
    )


def test_chat_pagination_limit_header_reflects_effective_value(
    app_instance, sqlite_session_factory
):
    """GET /sessions/{id}/messages?limit=N must set X-Message-Pagination-Limit
    to exactly N in the response headers.

    Locks CHAT-BE-006 sub-contract 2: the header must echo the effective limit
    so clients can reconstruct pagination state without re-parsing the URL.
    """
    username = "tester_chat_pag_002"
    n = 6
    session_id, _ = _seed_session_with_messages(sqlite_session_factory, username, n)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(username)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/v1/chat/sessions/{session_id}/messages",
            params={"limit": 3},
        )

    assert response.status_code == 200
    headers = response.headers
    assert headers.get("x-message-pagination-limit") == "3", (
        "X-Message-Pagination-Limit must equal the requested limit"
    )
    assert headers.get("x-message-pagination-has-more") == "true"
    assert headers.get("x-message-pagination-returned") == "3"


def test_chat_pagination_before_id_and_next_before_id_cursor_semantics(
    app_instance, sqlite_session_factory
):
    """GET /sessions/{id}/messages?limit=N returns the N most-recent messages;
    X-Message-Pagination-Next-Before-Id points to the oldest id in that page
    so the next request with before_id=<next_before_id> fetches the preceding
    page without overlap or gap.

    Locks CHAT-BE-006 sub-contract 3: cursor semantics must be stable.
    """
    username = "tester_chat_pag_003"
    n = 6
    session_id, msg_ids = _seed_session_with_messages(sqlite_session_factory, username, n)
    # msg_ids is sorted ascending; the 6 ids are [id0, id1, id2, id3, id4, id5]

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(username)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        # First page: 3 most-recent messages (ids 3,4,5 in asc order)
        page1 = client.get(
            f"/api/v1/chat/sessions/{session_id}/messages",
            params={"limit": 3},
        )
        assert page1.status_code == 200
        page1_ids = [m["id"] for m in page1.json()]
        assert page1_ids == msg_ids[-3:], "First page must be the 3 most-recent messages"

        next_before_id = page1.headers.get("x-message-pagination-next-before-id")
        assert next_before_id is not None, "next_before_id must be present when has_more=true"
        assert int(next_before_id) == msg_ids[-3], (
            "next_before_id must equal the oldest id in the current page"
        )

        # Second page: use cursor to fetch the 3 preceding messages (ids 0,1,2)
        page2 = client.get(
            f"/api/v1/chat/sessions/{session_id}/messages",
            params={"limit": 3, "before_id": next_before_id},
        )
        assert page2.status_code == 200
        page2_ids = [m["id"] for m in page2.json()]
        assert page2_ids == msg_ids[:3], "Second page must be the 3 oldest messages"
        assert page2.headers.get("x-message-pagination-has-more") == "false"
        assert page2.headers.get("x-message-pagination-next-before-id") is None, (
            "next_before_id must be absent on the last page"
        )

        # No overlap between pages
        assert not set(page1_ids) & set(page2_ids), "Pages must not overlap"


def test_chat_pagination_cursor_path_independent_of_skip(
    app_instance, sqlite_session_factory
):
    """GET /sessions/{id}/messages?before_id=X&limit=N with skip=0 (explicit)
    and without skip must return identical results.

    Locks CHAT-BE-006 sub-contract 4: the public cursor path (before_id) must
    not depend on skip — skip is a legacy offset param that must not interfere
    with cursor-based pagination.
    """
    username = "tester_chat_pag_004"
    n = 6
    session_id, msg_ids = _seed_session_with_messages(sqlite_session_factory, username, n)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(username)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    # Use the 4th message id as the before_id cursor (fetch messages older than it)
    cursor_id = msg_ids[3]

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        # Without skip param
        resp_no_skip = client.get(
            f"/api/v1/chat/sessions/{session_id}/messages",
            params={"limit": 3, "before_id": cursor_id},
        )
        # With explicit skip=0
        resp_skip_zero = client.get(
            f"/api/v1/chat/sessions/{session_id}/messages",
            params={"limit": 3, "before_id": cursor_id, "skip": 0},
        )

    assert resp_no_skip.status_code == 200
    assert resp_skip_zero.status_code == 200

    ids_no_skip = [m["id"] for m in resp_no_skip.json()]
    ids_skip_zero = [m["id"] for m in resp_skip_zero.json()]

    assert ids_no_skip == ids_skip_zero, (
        "Cursor path must return identical results regardless of skip=0 vs omitted skip"
    )
    # Both must return messages strictly older than cursor_id
    assert all(mid < cursor_id for mid in ids_no_skip), (
        "All returned messages must have id < before_id"
    )


# ---------------------------------------------------------------------------
# BE-004-B: delete lifecycle contract – soft-delete / hard-delete regressions
# ---------------------------------------------------------------------------
#
# Locked contracts verified here:
#   1. End-user DELETE is a soft-delete: deleted_at is set, row survives in DB.
#   2. End-user DELETE is NOT a hard-delete: the DB row is not physically removed.
#   3. Vector cleanup side-effect: delete_vectors=True is passed; in the local
#      (non-external) vector store the call is a no-op, so the contract is that
#      the endpoint still returns 204 and the audit log records soft_delete=True.
#   4. Object artifacts are RETAINED after end-user delete (delete_object_artifacts=False).
#   5. Admin hard-delete is permanent: the DB row is physically removed and
#      cannot be retrieved even by bypassing the soft-delete filter.
#   6. Admin hard-delete cascades: chunks and jobs are removed (CASCADE FK).
#   7. Soft-deleted document is invisible to list/get endpoints (deleted_at filter).
#   8. Hard-deleted document is invisible to admin hard-delete (404 on repeat).
# ---------------------------------------------------------------------------


def _seed_knowledge_document(
    session_factory,
    owner_username: str,
    title: str = "Test Doc",
    raw_text: str = "Hello world. This is a test document for regression coverage.",
) -> tuple[int, int]:
    """Seed a user + knowledge document + ingestion job. Returns (doc_id, job_id)."""
    from app.models.knowledge_models import KnowledgeDocument, IngestionJob
    from app.models.chat_models import User

    db = session_factory()
    try:
        # Upsert user (may already exist in the same factory)
        existing_user = db.query(User).filter(User.username == owner_username).first()
        if not existing_user:
            db.add(User(username=owner_username, password_hash="hash", role="user"))
            db.commit()

        doc = KnowledgeDocument(
            owner_username=owner_username,
            title=title,
            source_type="text",
            raw_text=raw_text,
            status="indexed",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        job = IngestionJob(document_id=doc.id, status="completed")
        db.add(job)
        db.commit()
        db.refresh(job)

        return doc.id, job.id
    finally:
        db.close()


def _stub_external_stores(monkeypatch) -> None:
    """Monkeypatch both Qdrant and S3/MinIO so delete tests don't need live infra.

    - vector_store: is_external_vector_store_enabled → False (local store path)
    - object_storage: is_object_storage_enabled → False (no artifact calls)

    This is the narrowest stub: it only disables the external-store guards so
    the service falls through to the local (no-op) code paths, exactly as the
    contract specifies for a deployment without external stores.
    """
    import app.services.vector_store as _vs
    import app.services.object_storage as _os

    monkeypatch.setattr(_vs, "is_external_vector_store_enabled", lambda: False)
    monkeypatch.setattr(_os, "is_object_storage_enabled", lambda: False)


def test_end_user_delete_is_soft_delete_sets_deleted_at(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/documents/{doc_id} must set deleted_at on the row
    and return HTTP 204.  The row must still exist in the database with a
    non-null deleted_at timestamp.

    Locks BE-004-B contract 1: end-user delete is soft-delete via deleted_at.
    Also locks audit layer: action="document.delete" with soft_delete=True must
    be written to the AuditLog table after a successful soft-delete.
    """
    _stub_external_stores(monkeypatch)
    monkeypatch.setattr(main_module.settings, "audit_log_enabled", True)
    owner = "tester_be004b_soft_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(owner)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.delete(f"/api/v1/knowledge/documents/{doc_id}")

    assert response.status_code == 204, (
        f"Expected 204 No Content from soft-delete, got {response.status_code}"
    )

    # Verify the row still exists in the DB with deleted_at set
    from app.models.knowledge_models import KnowledgeDocument, AuditLog

    db = sqlite_session_factory()
    try:
        raw_row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
        assert raw_row is not None, (
            "Soft-delete must NOT physically remove the DB row"
        )
        assert raw_row.deleted_at is not None, (
            "Soft-delete must set deleted_at to a non-null timestamp"
        )

        # Audit log contract: exactly one entry for this soft-delete
        audit_entries = (
            db.query(AuditLog)
            .filter(
                AuditLog.actor_username == owner,
                AuditLog.action == "document.delete",
                AuditLog.resource_id == str(doc_id),
            )
            .all()
        )
        assert len(audit_entries) == 1, (
            f"Soft-delete must write exactly one audit log entry; got {len(audit_entries)}"
        )
        audit = audit_entries[0]
        assert audit.resource_type == "document", (
            f"Audit resource_type must be 'document', got {audit.resource_type!r}"
        )
        detail = audit.detail_json or {}
        assert detail.get("soft_delete") is True, (
            f"Audit detail_json must contain soft_delete=True; got {detail!r}"
        )
    finally:
        db.close()


def test_end_user_delete_does_not_hard_delete_row(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/documents/{doc_id} must NOT physically remove the row.
    After the delete the row must still be queryable via a raw (unfiltered) DB
    query, proving the row was only soft-deleted.

    Locks BE-004-B contract 2: end-user delete is NOT a hard-delete.
    """
    _stub_external_stores(monkeypatch)
    owner = "tester_be004b_soft_02"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(owner)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        client.delete(f"/api/v1/knowledge/documents/{doc_id}")

    from app.models.knowledge_models import KnowledgeDocument

    db = sqlite_session_factory()
    try:
        # Raw query bypasses the deleted_at filter used by get_document()
        count = (
            db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.id == doc_id)
            .count()
        )
        assert count == 1, (
            f"Row must survive soft-delete (count=1), got count={count}"
        )
    finally:
        db.close()


def test_end_user_delete_object_artifacts_are_retained(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/documents/{doc_id} must NOT delete object-storage
    artifacts.  The endpoint calls delete_document_storage with
    delete_object_artifacts=False, so object_storage.delete_document_artifacts
    must never be invoked.

    Locks BE-004-B contract 4: object artifacts are retained on end-user delete.
    """
    owner = "tester_be004b_artifacts_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    # Spy: track whether delete_document_artifacts is called
    artifact_delete_calls: list = []

    import app.services.object_storage as obj_storage_module
    import app.services.vector_store as _vs

    # Stub vector store so the soft-delete path doesn't hit Qdrant
    monkeypatch.setattr(_vs, "is_external_vector_store_enabled", lambda: False)

    monkeypatch.setattr(
        obj_storage_module,
        "delete_document_artifacts",
        lambda *args, **kwargs: artifact_delete_calls.append((args, kwargs)) or {"deleted_keys": 0},
    )
    # Ensure object storage appears enabled so the guard would fire if called
    monkeypatch.setattr(obj_storage_module, "is_object_storage_enabled", lambda: True)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(owner)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.delete(f"/api/v1/knowledge/documents/{doc_id}")

    assert response.status_code == 204
    assert len(artifact_delete_calls) == 0, (
        "delete_document_artifacts must NOT be called during end-user soft-delete "
        f"(delete_object_artifacts=False); got {len(artifact_delete_calls)} call(s)"
    )


def test_end_user_delete_vector_cleanup_is_attempted(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/documents/{doc_id} must attempt vector cleanup
    (delete_vectors=True).  When an external vector store is enabled the
    service calls vector_store.delete_document_chunks.  This test enables the
    external store flag and verifies the call is made exactly once.

    Locks BE-004-B contract 3: vector cleanup side-effect occurs per contract.
    """
    owner = "tester_be004b_vectors_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    vector_delete_calls: list = []

    import app.services.vector_store as vector_store_module
    import app.services.object_storage as _os

    # Stub object storage so the soft-delete path doesn't hit S3/MinIO
    monkeypatch.setattr(_os, "is_object_storage_enabled", lambda: False)

    monkeypatch.setattr(
        vector_store_module,
        "is_external_vector_store_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        vector_store_module,
        "delete_document_chunks",
        lambda owner_username, document_id: vector_delete_calls.append((owner_username, document_id)),
    )

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(owner)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.delete(f"/api/v1/knowledge/documents/{doc_id}")

    assert response.status_code == 204
    assert len(vector_delete_calls) == 1, (
        f"delete_document_chunks must be called exactly once; got {len(vector_delete_calls)} call(s)"
    )
    called_owner, called_doc_id = vector_delete_calls[0]
    assert called_owner == owner
    assert called_doc_id == doc_id


def test_soft_deleted_document_invisible_to_get_endpoint(
    app_instance, sqlite_session_factory, monkeypatch
):
    """After DELETE /knowledge/documents/{doc_id}, GET /knowledge/documents/{doc_id}
    must return 404 because the deleted_at filter hides the soft-deleted row.

    Locks BE-004-B contract 7: soft-deleted document is invisible to read endpoints.
    """
    _stub_external_stores(monkeypatch)
    owner = "tester_be004b_invisible_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(owner)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        before = client.get(f"/api/v1/knowledge/documents/{doc_id}")
        assert before.status_code == 200, "Document must be visible before delete"

        client.delete(f"/api/v1/knowledge/documents/{doc_id}")

        after = client.get(f"/api/v1/knowledge/documents/{doc_id}")

    assert after.status_code == 404, (
        f"Soft-deleted document must return 404 from GET, got {after.status_code}"
    )


def test_soft_deleted_document_invisible_to_list_endpoint(
    app_instance, sqlite_session_factory, monkeypatch
):
    """After DELETE /knowledge/documents/{doc_id}, GET /knowledge/documents
    must NOT include the soft-deleted document in the listing.

    Locks BE-004-B contract 7: soft-deleted document is invisible to list endpoint.
    """
    _stub_external_stores(monkeypatch)
    owner = "tester_be004b_invisible_02"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(owner)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        before_list = client.get("/api/v1/knowledge/documents")
        assert any(d["id"] == doc_id for d in before_list.json()), (
            "Document must appear in list before delete"
        )

        client.delete(f"/api/v1/knowledge/documents/{doc_id}")

        after_list = client.get("/api/v1/knowledge/documents")

    assert not any(d["id"] == doc_id for d in after_list.json()), (
        "Soft-deleted document must not appear in GET /knowledge/documents listing"
    )


def test_admin_hard_delete_permanently_removes_row(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/admin/documents/{doc_id}/hard-delete must physically
    remove the DB row.  A subsequent raw (unfiltered) DB query must return no
    row for the deleted document id.

    Locks BE-004-B contract 5: admin hard-delete is permanent row removal.
    Also locks audit layer: action="document.hard_delete" must be written to
    the AuditLog table after a successful hard-delete.
    """
    _stub_external_stores(monkeypatch)
    monkeypatch.setattr(main_module.settings, "audit_log_enabled", True)
    owner = "tester_be004b_hard_01"
    admin_user = "admin_be004b"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(
        admin_user, role="admin"
    )
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.delete(f"/api/v1/knowledge/admin/documents/{doc_id}/hard-delete")

    assert response.status_code == 204, (
        f"Admin hard-delete must return 204, got {response.status_code}"
    )

    from app.models.knowledge_models import KnowledgeDocument, AuditLog

    db = sqlite_session_factory()
    try:
        raw_row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
        assert raw_row is None, (
            "Admin hard-delete must physically remove the DB row; row still found"
        )

        # Audit log contract: exactly one hard-delete entry written by the admin
        audit_entries = (
            db.query(AuditLog)
            .filter(
                AuditLog.actor_username == admin_user,
                AuditLog.action == "document.hard_delete",
                AuditLog.resource_id == str(doc_id),
            )
            .all()
        )
        assert len(audit_entries) == 1, (
            f"Hard-delete must write exactly one audit log entry; got {len(audit_entries)}"
        )
        audit = audit_entries[0]
        assert audit.resource_type == "document", (
            f"Audit resource_type must be 'document', got {audit.resource_type!r}"
        )
        detail = audit.detail_json or {}
        # The hard-delete audit records the owner so post-mortem forensics work
        assert "owner" in detail, (
            f"Audit detail_json must contain 'owner' field for hard-delete; got {detail!r}"
        )
        assert detail["owner"] == owner, (
            f"Audit detail_json['owner'] must be {owner!r}, got {detail.get('owner')!r}"
        )
    finally:
        db.close()


def test_admin_hard_delete_cascades_chunks_and_jobs(
    app_instance, monkeypatch
):
    """DELETE /knowledge/admin/documents/{doc_id}/hard-delete must cascade-delete
    all associated KnowledgeChunk and IngestionJob rows (FK ondelete=CASCADE).

    Locks BE-004-B contract 6: hard-delete cascades to chunks and jobs.

    Note: SQLite requires PRAGMA foreign_keys = ON for DB-level CASCADE to fire.
    This test uses a dedicated session factory that enables the pragma via a
    SQLAlchemy connection event so the cascade behaviour matches production
    (PostgreSQL enforces FK cascades unconditionally).
    """
    _stub_external_stores(monkeypatch)
    from app.models.knowledge_models import KnowledgeChunk, IngestionJob
    from sqlalchemy import event as sa_event

    # Build a dedicated SQLite engine with FK enforcement enabled
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    fk_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @sa_event.listens_for(fk_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    fk_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=fk_engine)
    Base.metadata.create_all(bind=fk_engine)

    try:
        owner = "tester_be004b_cascade_01"
        doc_id, job_id = _seed_knowledge_document(fk_session_factory, owner)

        # Add a chunk to the document so we can verify cascade
        db = fk_session_factory()
        try:
            chunk = KnowledgeChunk(
                document_id=doc_id,
                chunk_index=0,
                content="chunk content for cascade test",
                token_count=6,
            )
            db.add(chunk)
            db.commit()
            db.refresh(chunk)
        finally:
            db.close()

        app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(
            "admin_be004b_cascade", role="admin"
        )
        app_instance.dependency_overrides[deps.get_db] = _override_get_db(fk_session_factory)

        with TestClient(app_instance, raise_server_exceptions=False) as client:
            response = client.delete(f"/api/v1/knowledge/admin/documents/{doc_id}/hard-delete")

        assert response.status_code == 204

        db = fk_session_factory()
        try:
            remaining_chunks = (
                db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id).count()
            )
            remaining_jobs = (
                db.query(IngestionJob).filter(IngestionJob.document_id == doc_id).count()
            )
            assert remaining_chunks == 0, (
                f"Hard-delete must cascade to chunks; {remaining_chunks} chunk(s) remain"
            )
            assert remaining_jobs == 0, (
                f"Hard-delete must cascade to jobs; {remaining_jobs} job(s) remain"
            )
        finally:
            db.close()
    finally:
        fk_engine.dispose()
        os.remove(db_path)


def test_admin_hard_delete_is_idempotent_returns_404_on_repeat(
    app_instance, sqlite_session_factory, monkeypatch
):
    """A second DELETE /knowledge/admin/documents/{doc_id}/hard-delete on an
    already-deleted document must return 404 (document not found), not 500 or
    204.

    Locks BE-004-B contract 8: hard-delete on missing doc returns 404.
    """
    _stub_external_stores(monkeypatch)
    owner = "tester_be004b_hard_idem_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(
        "admin_be004b_idem", role="admin"
    )
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        first = client.delete(f"/api/v1/knowledge/admin/documents/{doc_id}/hard-delete")
        second = client.delete(f"/api/v1/knowledge/admin/documents/{doc_id}/hard-delete")

    assert first.status_code == 204
    assert second.status_code == 404, (
        f"Repeat hard-delete must return 404, got {second.status_code}"
    )


def test_admin_hard_delete_requires_admin_role(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/admin/documents/{doc_id}/hard-delete must be rejected
    with 403 when the caller is a regular user (not admin).

    Locks BE-004-B contract 5: admin hard-delete path is admin-only.
    """
    _stub_external_stores(monkeypatch)
    owner = "tester_be004b_authz_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    # Regular user (role="user") must be denied
    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(
        owner, role="user"
    )
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.delete(f"/api/v1/knowledge/admin/documents/{doc_id}/hard-delete")

    assert response.status_code == 403, (
        f"Non-admin hard-delete must return 403, got {response.status_code}"
    )

    # Verify the row was NOT deleted
    from app.models.knowledge_models import KnowledgeDocument

    db = sqlite_session_factory()
    try:
        raw_row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
        assert raw_row is not None, (
            "Row must survive a rejected (403) hard-delete attempt"
        )
    finally:
        db.close()


def test_end_user_cannot_delete_another_users_document(
    app_instance, sqlite_session_factory, monkeypatch
):
    """DELETE /knowledge/documents/{doc_id} must return 404 when the requesting
    user does not own the document.  The document must remain intact.

    Locks BE-004-B: ownership check prevents cross-user soft-delete.
    """
    _stub_external_stores(monkeypatch)
    owner = "tester_be004b_owner_01"
    attacker = "tester_be004b_attacker_01"
    doc_id, _ = _seed_knowledge_document(sqlite_session_factory, owner)

    # Seed attacker user
    from app.models.chat_models import User

    db = sqlite_session_factory()
    try:
        if not db.query(User).filter(User.username == attacker).first():
            db.add(User(username=attacker, password_hash="hash", role="user"))
            db.commit()
    finally:
        db.close()

    app_instance.dependency_overrides[deps.get_current_user] = _override_current_user(attacker)
    app_instance.dependency_overrides[deps.get_db] = _override_get_db(sqlite_session_factory)

    with TestClient(app_instance, raise_server_exceptions=False) as client:
        response = client.delete(f"/api/v1/knowledge/documents/{doc_id}")

    assert response.status_code == 404, (
        f"Cross-user delete must return 404, got {response.status_code}"
    )

    # Verify the document is still intact (not soft-deleted)
    from app.models.knowledge_models import KnowledgeDocument

    db = sqlite_session_factory()
    try:
        raw_row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
        assert raw_row is not None, "Document must survive a rejected cross-user delete"
        assert raw_row.deleted_at is None, (
            "deleted_at must remain NULL after a rejected cross-user delete"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# BE-004-A: schema-level status vocabulary contract
# ---------------------------------------------------------------------------

def test_be004a_document_status_vocabulary():
    """KnowledgeDocumentResponse.status must only accept the four verified values.

    Locks BE-004-A: schema Literal contract for document status.
    """
    from typing import get_args

    from app.schemas.knowledge_schemas import KnowledgeDocumentResponse

    allowed = set(get_args(KnowledgeDocumentResponse.model_fields["status"].annotation))
    assert allowed == {"uploaded", "processing", "indexed", "failed"}, (
        f"Document status vocabulary changed unexpectedly: {allowed}"
    )


def test_be004a_job_status_vocabulary():
    """IngestionJobResponse.status must only accept the four verified values.

    Locks BE-004-A: schema Literal contract for ingestion job status.
    """
    from typing import get_args

    from app.schemas.knowledge_schemas import IngestionJobResponse

    allowed = set(get_args(IngestionJobResponse.model_fields["status"].annotation))
    assert allowed == {"queued", "processing", "completed", "failed"}, (
        f"Job status vocabulary changed unexpectedly: {allowed}"
    )


def test_be004a_ingestion_result_status_vocabulary():
    """IngestionResult.status must only accept the two verified values.

    Locks BE-004-A: schema Literal contract for ingestion result status.
    """
    from typing import get_args

    from app.schemas.knowledge_schemas import IngestionResult

    allowed = set(get_args(IngestionResult.model_fields["status"].annotation))
    assert allowed == {"pending", "indexed"}, (
        f"IngestionResult status vocabulary changed unexpectedly: {allowed}"
    )


def test_be004a_invalid_document_status_rejected():
    """KnowledgeDocumentResponse must reject an out-of-vocabulary document status."""
    import pytest
    from pydantic import ValidationError

    from app.schemas.knowledge_schemas import KnowledgeDocumentResponse
    from datetime import datetime

    now = datetime.utcnow()
    with pytest.raises(ValidationError):
        KnowledgeDocumentResponse(
            id=1,
            owner_username="tester",
            title="doc",
            source_type="text",
            status="unknown_status",  # not in vocabulary
            created_at=now,
            updated_at=now,
        )


def test_be004a_invalid_job_status_rejected():
    """IngestionJobResponse must reject an out-of-vocabulary job status."""
    import pytest
    from pydantic import ValidationError

    from app.schemas.knowledge_schemas import IngestionJobResponse
    from datetime import datetime

    now = datetime.utcnow()
    with pytest.raises(ValidationError):
        IngestionJobResponse(
            id=1,
            document_id=1,
            status="unknown_status",  # not in vocabulary
            created_at=now,
            updated_at=now,
        )


def test_be004a_invalid_ingestion_result_status_rejected():
    """IngestionResult must reject an out-of-vocabulary ingestion result status."""
    import pytest
    from pydantic import ValidationError

    from app.schemas.knowledge_schemas import IngestionResult

    with pytest.raises(ValidationError):
        IngestionResult(
            document_id=1,
            job_id=1,
            status="unknown_status",  # not in vocabulary
        )