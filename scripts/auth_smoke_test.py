from pathlib import Path
import sys

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.deps import get_db
from app.core.database import Base
from app.main import app
from app.models.chat_models import User


def main() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        debug_env_response = client.get("/debug/env")
        assert debug_env_response.status_code == 404, debug_env_response.text

        register_payload = {
            "username": "phase1_user",
            "password": "StrongPass1!",
            "confirm_password": "StrongPass1!",
        }
        register_response = client.post("/api/auth/register", json=register_payload)
        assert register_response.status_code == 201, register_response.text

        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me_response = client.get("/api/auth/me", headers=headers)
        assert me_response.status_code == 200, me_response.text
        assert me_response.json()["username"] == "phase1_user"

        register_target_response = client.post(
            "/api/auth/register",
            json={
                "username": "phase1_target",
                "password": "TargetPass1!",
                "confirm_password": "TargetPass1!",
            },
        )
        assert register_target_response.status_code == 201, register_target_response.text

        seed_test_user_response = client.post(
            "/api/auth/register",
            json={
                "username": "test_user",
                "password": "StrongPass1!",
                "confirm_password": "StrongPass1!",
            },
        )
        assert seed_test_user_response.status_code == 201, seed_test_user_response.text
        assert seed_test_user_response.json()["role"] == "admin"

        promote_db = testing_session_local()
        try:
            phase1_user = promote_db.query(User).filter(User.username == "phase1_user").first()
            assert phase1_user is not None
            phase1_user.role = "admin"

            test_user = promote_db.query(User).filter(User.username == "test_user").first()
            assert test_user is not None
            test_user.role = "user"
            promote_db.commit()
        finally:
            promote_db.close()

        phase1_admin_attempt = client.get("/api/auth/admin/users", headers=headers)
        assert phase1_admin_attempt.status_code == 403, phase1_admin_attempt.text

        test_user_login_response = client.post(
            "/api/auth/login",
            json={"username": "test_user", "password": "StrongPass1!"},
        )
        assert test_user_login_response.status_code == 200, test_user_login_response.text
        assert test_user_login_response.json()["role"] == "admin"

        hardcoded_headers = {
            "Authorization": f"Bearer {test_user_login_response.json()['access_token']}"
        }
        hardcoded_me_response = client.get("/api/auth/me", headers=hardcoded_headers)
        assert hardcoded_me_response.status_code == 200, hardcoded_me_response.text
        assert hardcoded_me_response.json()["role"] == "admin"

        hardcoded_users_response = client.get("/api/auth/admin/users", headers=hardcoded_headers)
        assert hardcoded_users_response.status_code == 200, hardcoded_users_response.text
        assert any(row["username"] == "test_user" and row["role"] == "admin" for row in hardcoded_users_response.json())
        assert any(row["username"] == "phase1_user" and row["role"] == "user" for row in hardcoded_users_response.json())

        demote_hardcoded_response = client.post(
            "/api/auth/admin/set-role",
            json={"username": "test_user", "role": "user"},
            headers=hardcoded_headers,
        )
        assert demote_hardcoded_response.status_code == 200, demote_hardcoded_response.text
        assert demote_hardcoded_response.json()["success"] is True
        assert "admin" in demote_hardcoded_response.json()["message"]

        promote_phase1_response = client.post(
            "/api/auth/admin/set-role",
            json={"username": "phase1_user", "role": "admin"},
            headers=hardcoded_headers,
        )
        assert promote_phase1_response.status_code == 200, promote_phase1_response.text
        assert "user" in promote_phase1_response.json()["message"]

        hardcoded_users_response_after_demote = client.get("/api/auth/admin/users", headers=hardcoded_headers)
        assert hardcoded_users_response_after_demote.status_code == 200, hardcoded_users_response_after_demote.text
        assert any(
            row["username"] == "test_user" and row["role"] == "admin"
            for row in hardcoded_users_response_after_demote.json()
        )
        assert any(
            row["username"] == "phase1_user" and row["role"] == "user"
            for row in hardcoded_users_response_after_demote.json()
        )

        reset_token_response = client.post(
            "/api/auth/admin/reset-password",
            json={"username": "phase1_target", "expire_minutes": 30},
            headers=hardcoded_headers,
        )
        assert reset_token_response.status_code == 200, reset_token_response.text
        reset_token = reset_token_response.json()["reset_token"]

        public_reset_response = client.post(
            "/api/auth/reset-password",
            json={
                "username": "phase1_target",
                "reset_token": reset_token,
                "new_password": "ResetStrong3#",
                "confirm_new_password": "ResetStrong3#",
            },
        )
        assert public_reset_response.status_code == 200, public_reset_response.text

        target_login_after_reset = client.post(
            "/api/auth/login",
            json={"username": "phase1_target", "password": "ResetStrong3#"},
        )
        assert target_login_after_reset.status_code == 200, target_login_after_reset.text

        login_response = client.post(
            "/api/auth/login",
            json={"username": "phase1_user", "password": "StrongPass1!"},
        )
        assert login_response.status_code == 200, login_response.text
        auth_headers = {
            "Authorization": f"Bearer {login_response.json()['access_token']}"
        }

        # Test change password
        change_pw_response = client.post(
            "/api/auth/change-password",
            json={
                "old_password": "StrongPass1!",
                "new_password": "NewStrong2@",
                "confirm_new_password": "NewStrong2@",
            },
            headers=auth_headers,
        )
        assert change_pw_response.status_code == 200, change_pw_response.text

        # Login with new password
        login2 = client.post(
            "/api/auth/login",
            json={"username": "phase1_user", "password": "NewStrong2@"},
        )
        assert login2.status_code == 200, login2.text
        auth_headers = {"Authorization": f"Bearer {login2.json()['access_token']}"}

        session_response = client.post(
            "/api/chat/sessions",
            json={"title": "Chat 1"},
            headers=auth_headers,
        )
        assert session_response.status_code == 200, session_response.text
        session_id = session_response.json()["id"]

        list_sessions_response = client.get("/api/chat/sessions", headers=auth_headers)
        assert list_sessions_response.status_code == 200, list_sessions_response.text
        assert len(list_sessions_response.json()) == 1

        session_messages_response = client.get(
            f"/api/chat/sessions/{session_id}/messages",
            headers=auth_headers,
        )
        assert session_messages_response.status_code == 200, session_messages_response.text
        assert session_messages_response.json() == []

        usage_response = client.get("/api/chat/usage/me", headers=auth_headers)
        assert usage_response.status_code == 200, usage_response.text
        assert usage_response.json()["username"] == "phase1_user"

        weak_password_response = client.post(
            "/api/auth/register",
            json={
                "username": "weak_user",
                "password": "weak",
                "confirm_password": "weak",
            },
        )
        assert weak_password_response.status_code == 400, weak_password_response.text

    app.dependency_overrides.clear()
    print("AUTH_API_SMOKE_OK")


if __name__ == "__main__":
    main()

