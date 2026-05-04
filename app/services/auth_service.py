"""Authentication service – separated from chat_service."""
from sqlalchemy.orm import Session

from app.core.security import create_auth_token_pair, decode_refresh_token, normalize_username
from app.crud import crud_auth


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _build_auth_response(username: str, role: str = "user", token_version: int = 0):
    normalized = normalize_username(username)
    tokens = create_auth_token_pair(normalized, token_version=token_version)
    return {
        "success": True,
        "username": normalized,
        "role": role,
        **tokens,
        "token_type": "bearer",
    }


# ---- public API -----------------------------------------------------------

def register_user(db: Session, username: str, password: str):
    user = crud_auth.create_user(db, username=username, password=password)
    return _build_auth_response(
        user.username,
        user.role,
        token_version=int(getattr(user, "auth_token_version", 0) or 0),
    )


def login_user(db: Session, username: str, password: str):
    user = crud_auth.verify_user_credentials(db, username, password)
    if not user:
        raise ValueError("Invalid username or password.")
    return _build_auth_response(
        user.username,
        user.role,
        token_version=int(getattr(user, "auth_token_version", 0) or 0),
    )


def get_me(user):
    return {"username": user.username, "role": user.role}


def refresh_auth_tokens(db: Session, refresh_token: str):
    token_payload = decode_refresh_token(refresh_token)
    if not token_payload:
        raise ValueError("Invalid or expired refresh token.")

    user = crud_auth.get_user_by_username(db, token_payload["username"])
    if not user:
        raise ValueError("Authenticated user does not exist.")

    current_version = int(getattr(user, "auth_token_version", 0) or 0)
    if current_version != token_payload["token_version"]:
        raise ValueError("Refresh token has been revoked.")

    return _build_auth_response(user.username, user.role, token_version=current_version)


def logout_user(db: Session, user):
    crud_auth.revoke_auth_tokens(db, user)
    return {"success": True, "message": "Logged out successfully."}


def change_password(db: Session, user, old_password: str, new_password: str):
    crud_auth.change_password(db, user, old_password, new_password)
    return {"success": True, "message": "Password changed successfully."}


def admin_reset_password(db: Session, target_username: str, expire_minutes: int = 30):
    """Admin generates a reset token for a user."""
    user = crud_auth.get_user_by_username(db, target_username)
    if not user:
        raise ValueError(f"User '{target_username}' not found.")
    token = crud_auth.create_reset_token(db, user, expire_minutes)
    return {"username": user.username, "reset_token": token, "expire_minutes": expire_minutes}


def consume_reset_token(db: Session, username: str, token: str, new_password: str):
    crud_auth.consume_reset_token(db, username, token, new_password)
    return {"success": True, "message": "Password has been reset."}


def set_user_role(db: Session, target_username: str, role: str):
    user = crud_auth.get_user_by_username(db, target_username)
    if not user:
        raise ValueError(f"User '{target_username}' not found.")
    updated_user = crud_auth.set_user_role(db, user, role)
    return {"username": updated_user.username, "role": updated_user.role}


def list_users(db: Session, skip: int = 0, limit: int = 100):
    users = crud_auth.list_users(db, skip, limit)
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "max_tokens_per_day": u.max_tokens_per_day,
            "created_at": u.created_at,
        }
        for u in users
    ]

