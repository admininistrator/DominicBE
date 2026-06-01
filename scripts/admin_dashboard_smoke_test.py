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
from app.core.config import settings
from app.core.database import Base
from app.crud import crud_chat
from app.main import app
from app.models.chat_models import User


API_PREFIX = "/api/v1"


def api_path(path: str) -> str:
    return f"{API_PREFIX}{path}"


def _register(client: TestClient, username: str, password: str = "StrongPass1!") -> dict:
    response = client.post(
        api_path("/auth/register"),
        json={
            "username": username,
            "password": password,
            "confirm_password": password,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


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
    original_rate_limit_enabled = settings.rate_limit_enabled
    original_rag_core_mode = settings.rag_core_mode
    settings.rate_limit_enabled = False
    settings.rag_core_mode = "library"

    try:
        with TestClient(app) as client:
            user_payload = _register(client, "admin_dash_user")
            admin_payload = _register(client, "admin_dash_admin")
            user_headers = {"Authorization": f"Bearer {user_payload['access_token']}"}
            admin_headers = {"Authorization": f"Bearer {admin_payload['access_token']}"}

            promote_db = testing_session_local()
            try:
                admin_user = promote_db.query(User).filter(User.username == "admin_dash_admin").first()
                assert admin_user is not None
                admin_user.role = "admin"
                promote_db.commit()
            finally:
                promote_db.close()

            forbidden_response = client.get(api_path("/admin/overview"), headers=user_headers)
            assert forbidden_response.status_code == 403, forbidden_response.text

            session_response = client.post(
                api_path("/chat/sessions"),
                json={"title": "Private debug session"},
                headers=user_headers,
            )
            assert session_response.status_code == 200, session_response.text
            session_id = session_response.json()["id"]

            message_db = testing_session_local()
            try:
                crud_chat.create_message(
                    message_db,
                    role="user",
                    sender_username="admin_dash_user",
                    session_id=session_id,
                    content="This message content must not appear in admin session metadata.",
                    request_id="admin-dashboard-smoke-request",
                    input_tokens=7,
                    output_tokens=0,
                    status="success",
                )
            finally:
                message_db.close()

            overview_response = client.get(api_path("/admin/overview"), headers=admin_headers)
            assert overview_response.status_code == 200, overview_response.text
            overview = overview_response.json()
            assert "counts" in overview
            assert overview["health"]["dependencies"]["rag_core"]["status"] == "library_mode"

            health_response = client.get(api_path("/admin/health"), headers=admin_headers)
            assert health_response.status_code == 200, health_response.text
            assert "rag_core" in health_response.json()["dependencies"]

            settings_response = client.get(api_path("/admin/settings"), headers=admin_headers)
            assert settings_response.status_code == 200, settings_response.text
            settings_payload = settings_response.json()
            serialized_settings = str(settings_payload)
            assert "change-this-in-production" not in serialized_settings
            assert "password_hash" not in serialized_settings
            assert settings_payload["rag_core"]["api_key_configured"] is False

            sessions_response = client.get(api_path("/admin/sessions"), headers=admin_headers)
            assert sessions_response.status_code == 200, sessions_response.text
            sessions_payload = sessions_response.json()
            assert sessions_payload["total"] >= 1
            serialized_sessions = str(sessions_payload)
            assert "This message content must not appear" not in serialized_sessions
            assert "content" not in sessions_payload["items"][0]

            bad_delete_response = client.request(
                "DELETE",
                api_path(f"/admin/sessions/{session_id}"),
                json={"confirm": "WRONG"},
                headers=admin_headers,
            )
            assert bad_delete_response.status_code == 400, bad_delete_response.text

            delete_response = client.request(
                "DELETE",
                api_path(f"/admin/sessions/{session_id}"),
                json={"confirm": "DELETE_SESSION"},
                headers=admin_headers,
            )
            assert delete_response.status_code == 200, delete_response.text
            assert delete_response.json()["success"] is True

            audit_response = client.get(
                api_path("/knowledge/admin/audit-logs"),
                headers=admin_headers,
            )
            assert audit_response.status_code == 200, audit_response.text
            assert any(row["action"] == "admin.session.delete" for row in audit_response.json())

        print("ADMIN_DASHBOARD_SMOKE_OK")
    finally:
        settings.rate_limit_enabled = original_rate_limit_enabled
        settings.rag_core_mode = original_rag_core_mode
        app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
