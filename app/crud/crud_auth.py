"""CRUD operations for authentication & user management."""
from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe
from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import (
    hash_password,
    normalize_password,
    normalize_username,
    password_hash_needs_update,
    validate_username_policy,
    verify_password,
)
from app.models.chat_models import User


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _next_auth_token_version(user: User) -> int:
    current_version = int(getattr(user, "auth_token_version", 0) or 0)
    return current_version + 1


# ---------------------------------------------------------------------------
# User lookup
# ---------------------------------------------------------------------------

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    normalized = normalize_username(username)
    if not normalized:
        return None
    return db.query(User).filter(User.username == normalized).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def list_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).order_by(User.id.asc()).offset(skip).limit(limit).all()


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
    normalized = validate_username_policy(username, field_name="Username", min_length=3, max_length=255)

    if get_user_by_username(db, normalized):
        raise ValueError("Username already exists.")

    if role not in ("user", "admin"):
        raise ValueError("Role must be 'user' or 'admin'.")

    user = User(
        username=normalized,
        password_hash=hash_password(password, enforce_policy=True),
        role=role,
        auth_token_version=0,
        max_tokens_per_day=max_tokens_per_day,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Credential verification (with legacy migration)
# ---------------------------------------------------------------------------

def verify_user_credentials(db: Session, username: str, password: str) -> Optional[User]:
    try:
        normalized = validate_username_policy(username, field_name="Username", min_length=1, max_length=255)
    except ValueError:
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
        return user

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
    user.auth_token_version = _next_auth_token_version(user)
    db.commit()
    db.refresh(user)
    return user


def revoke_auth_tokens(db: Session, user: User) -> User:
    user.auth_token_version = _next_auth_token_version(user)
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
    user.reset_token_expires_at = _utcnow_naive() + timedelta(minutes=expire_minutes)
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

    if user.reset_token_expires_at and user.reset_token_expires_at < _utcnow_naive():
        user.reset_token = None
        user.reset_token_expires_at = None
        db.commit()
        raise ValueError("Reset token has expired.")

    user.password_hash = hash_password(new_password, enforce_policy=True)
    user.auth_token_version = _next_auth_token_version(user)
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
    if user.role != role:
        user.auth_token_version = _next_auth_token_version(user)
    user.role = role
    db.commit()
    db.refresh(user)
    return user

