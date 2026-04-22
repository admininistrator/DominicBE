from datetime import UTC, datetime, timedelta

import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def normalize_username(username: str | None) -> str:
    return (username or "").strip()


def normalize_password(password: str | None) -> str:
    return (password or "").strip()


def validate_password_policy(password: str) -> str:
    raw = normalize_password(password)
    if not raw:
        raise ValueError("Password must not be empty.")

    if len(raw) < settings.auth_password_min_length:
        raise ValueError(
            f"Password must be at least {settings.auth_password_min_length} characters long."
        )
    if len(raw) > settings.auth_password_max_length:
        raise ValueError(
            f"Password must be at most {settings.auth_password_max_length} characters long."
        )
    if len(raw.encode("utf-8")) > 72:
        raise ValueError("Password is too long for bcrypt. Use at most 72 UTF-8 bytes.")

    has_lower = any(char.islower() for char in raw)
    has_upper = any(char.isupper() for char in raw)
    has_digit = any(char.isdigit() for char in raw)
    has_special = any(not char.isalnum() for char in raw)

    if not (has_lower and has_upper and has_digit and has_special):
        raise ValueError(
            "Password must include at least 1 lowercase letter, 1 uppercase letter, 1 digit, and 1 special character."
        )

    return raw


def hash_password(password: str, *, enforce_policy: bool = False) -> str:
    raw = validate_password_policy(password) if enforce_policy else normalize_password(password)
    if not raw:
        raise ValueError("Password must not be empty.")
    return pwd_context.hash(raw)


def verify_password(plain_password: str, password_hash: str) -> bool:
    raw = normalize_password(plain_password)
    if not raw or not password_hash:
        return False
    try:
        return pwd_context.verify(raw, password_hash)
    except ValueError:
        return False


def password_hash_needs_update(password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return pwd_context.needs_update(password_hash)
    except ValueError:
        return False


def create_access_token(username: str) -> str:
    subject = normalize_username(username)
    if not subject:
        raise ValueError("Username must not be empty.")

    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(
            (now + timedelta(minutes=settings.auth_access_token_expire_minutes)).timestamp()
        ),
    }
    return jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm)


def decode_access_token(token: str | None) -> str | None:
    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            settings.auth_secret_key,
            algorithms=[settings.auth_algorithm],
        )
    except InvalidTokenError:
        return None

    if payload.get("type") != "access":
        return None

    subject = normalize_username(payload.get("sub"))
    return subject or None

