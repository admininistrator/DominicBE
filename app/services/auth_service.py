"""Authentication service – separated from chat_service."""
from sqlalchemy.orm import Session

from app.core.security import create_access_token, normalize_username
from app.crud import crud_auth


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _build_auth_response(username: str, role: str = "user"):
    normalized = normalize_username(username)
    return {
        "success": True,
        "username": normalized,
        "role": role,
        "access_token": create_access_token(normalized),
        "token_type": "bearer",
    }


# ---- public API -----------------------------------------------------------

def register_user(db: Session, username: str, password: str):
    user = crud_auth.create_user(db, username=username, password=password)
    return _build_auth_response(user.username, user.role)


def login_user(db: Session, username: str, password: str):
    user = crud_auth.verify_user_credentials(db, username, password)
    if not user:
        raise ValueError("Invalid username or password.")
    return _build_auth_response(user.username, user.role)


def get_me(user):
    return {"username": user.username, "role": user.role}


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

