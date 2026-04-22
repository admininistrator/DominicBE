"""CRUD operations for authentication & user management."""
from datetime import datetime, timedelta
from secrets import token_urlsafe
from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import (
    hash_password,
    normalize_password,
    normalize_username,
    password_hash_needs_update,
    verify_password,
)
from app.models.chat_models import User


HARDCODED_ADMIN_USERNAME = "test_user"


def _is_hardcoded_admin_username(username: str | None) -> bool:
    return normalize_username(username or "") == HARDCODED_ADMIN_USERNAME


def is_effective_admin_username(username: str | None) -> bool:
    return _is_hardcoded_admin_username(username)


def _apply_hardcoded_admin_role(db: Session, user: User | None) -> User | None:
    if not user:
        return None
    effective_role = "admin" if _is_hardcoded_admin_username(user.username) else "user"
    if user.role != effective_role:
        user.role = effective_role
        db.commit()
        db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# User lookup
# ---------------------------------------------------------------------------

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    normalized = normalize_username(username)
    if not normalized:
        return None
    user = db.query(User).filter(User.username == normalized).first()
    return _apply_hardcoded_admin_role(db, user)


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    user = db.query(User).filter(User.id == user_id).first()
    return _apply_hardcoded_admin_role(db, user)


def list_users(db: Session, skip: int = 0, limit: int = 100):
    users = db.query(User).order_by(User.id.asc()).offset(skip).limit(limit).all()
    return [_apply_hardcoded_admin_role(db, user) for user in users]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def create_user(
    db: Session,
    username: str,
    password: str,
    role: str = "user",
    max_tokens_per_day: int = 10000,
) -> User:
    normalized = normalize_username(username)
    if not normalized:
        raise ValueError("Username must not be empty.")

    if get_user_by_username(db, normalized):
        raise ValueError("Username already exists.")

    effective_role = "admin" if _is_hardcoded_admin_username(normalized) else "user"

    user = User(
        username=normalized,
        password_hash=hash_password(password, enforce_policy=True),
        role=effective_role,
        max_tokens_per_day=max_tokens_per_day,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _apply_hardcoded_admin_role(db, user)


# ---------------------------------------------------------------------------
# Credential verification (with legacy migration)
# ---------------------------------------------------------------------------

def verify_user_credentials(db: Session, username: str, password: str) -> Optional[User]:
    normalized = normalize_username(username)
    if not normalized:
        return None

    user = get_user_by_username(db, normalized)
    if not user:
        return None

    normalized_pw = normalize_password(password)
    if not normalized_pw:
        return None

    stored_hash = (user.password_hash or "").strip()

    # Verify bcrypt hash
    if stored_hash and verify_password(normalized_pw, stored_hash):
        if password_hash_needs_update(stored_hash):
            user.password_hash = hash_password(normalized_pw)
            db.commit()
            db.refresh(user)
        return _apply_hardcoded_admin_role(db, user)

    return None


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------

def change_password(db: Session, user: User, old_password: str, new_password: str) -> User:
    """Change password for authenticated user. Validates old password first."""
    normalized_old = normalize_password(old_password)
    stored_hash = (user.password_hash or "").strip()

    if not stored_hash or not verify_password(normalized_old, stored_hash):
        raise ValueError("Current password is incorrect.")

    user.password_hash = hash_password(new_password, enforce_policy=True)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Password reset (admin-issued token flow)
# ---------------------------------------------------------------------------

def create_reset_token(db: Session, user: User, expire_minutes: int = 30) -> str:
    """Generate a one-time reset token for a user (admin action)."""
    token = token_urlsafe(32)
    user.reset_token = token
    user.reset_token_expires_at = datetime.utcnow() + timedelta(minutes=expire_minutes)
    db.commit()
    db.refresh(user)
    return token


def consume_reset_token(db: Session, username: str, token: str, new_password: str) -> User:
    """Validate reset token and set new password."""
    user = get_user_by_username(db, username)
    if not user:
        raise ValueError("User not found.")

    if not user.reset_token or user.reset_token != token:
        raise ValueError("Invalid reset token.")

    if user.reset_token_expires_at and user.reset_token_expires_at < datetime.utcnow():
        user.reset_token = None
        user.reset_token_expires_at = None
        db.commit()
        raise ValueError("Reset token has expired.")

    user.password_hash = hash_password(new_password, enforce_policy=True)
    user.reset_token = None
    user.reset_token_expires_at = None
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Role management
# ---------------------------------------------------------------------------

def set_user_role(db: Session, user: User, role: str) -> User:
    if role not in ("user", "admin"):
        raise ValueError("Role must be 'user' or 'admin'.")
    user.role = "admin" if _is_hardcoded_admin_username(user.username) else "user"
    db.commit()
    db.refresh(user)
    return _apply_hardcoded_admin_role(db, user)

