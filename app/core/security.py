import string
from datetime import UTC, datetime, timedelta
from typing import TypedDict
from uuid import uuid4

import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SAFE_USERNAME_CHARACTERS = frozenset(string.ascii_letters + string.digits + "._-")
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


class DecodedTokenPayload(TypedDict):
    username: str
    token_version: int


def normalize_username(username: str | None) -> str:
    return (username or "").strip()


def validate_username_policy(
    username: str | None,
    *,
    field_name: str = "Username",
    min_length: int = 1,
    max_length: int = 255,
    allow_empty: bool = False,
) -> str | None:
    normalized = normalize_username(username)
    if allow_empty and not normalized:
        return None
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters long.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters long.")
    if normalized[0] in "._-" or normalized[-1] in "._-":
        raise ValueError(f"{field_name} must start and end with a letter or digit.")
    if any(character.isspace() for character in normalized):
        raise ValueError(f"{field_name} must not contain whitespace.")
    if any(character not in SAFE_USERNAME_CHARACTERS for character in normalized):
        raise ValueError(
            f"{field_name} may only contain letters, digits, dots, underscores, and hyphens."
        )
    return normalized


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


def _create_token(
    username: str,
    *,
    token_type: str,
    expire_minutes: int,
    token_version: int = 0,
) -> str:
    subject = normalize_username(username)
    if not subject:
        raise ValueError("Username must not be empty.")

    if token_version < 0:
        raise ValueError("Token version must not be negative.")

    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "type": token_type,
        "ver": int(token_version),
        "jti": uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm)


def create_access_token(username: str, token_version: int = 0) -> str:
    return _create_token(
        username,
        token_type=ACCESS_TOKEN_TYPE,
        expire_minutes=settings.auth_access_token_expire_minutes,
        token_version=token_version,
    )


def create_refresh_token(username: str, token_version: int = 0) -> str:
    return _create_token(
        username,
        token_type=REFRESH_TOKEN_TYPE,
        expire_minutes=settings.auth_refresh_token_expire_minutes,
        token_version=token_version,
    )


def create_auth_token_pair(username: str, token_version: int = 0) -> dict[str, str]:
    return {
        "access_token": create_access_token(username, token_version=token_version),
        "refresh_token": create_refresh_token(username, token_version=token_version),
    }


def _decode_token(
    token: str | None,
    *,
    expected_type: str,
) -> DecodedTokenPayload | None:
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

    if payload.get("type") != expected_type:
        return None

    subject_raw = payload.get("sub")
    if not isinstance(subject_raw, str):
        return None

    token_version_raw = payload.get("ver", 0)
    try:
        token_version = int(token_version_raw)
    except (TypeError, ValueError):
        return None

    if token_version < 0:
        return None

    subject = normalize_username(subject_raw)
    if not subject:
        return None

    return {"username": subject, "token_version": token_version}


def decode_access_token(token: str | None) -> DecodedTokenPayload | None:
    return _decode_token(token, expected_type=ACCESS_TOKEN_TYPE)


def decode_refresh_token(token: str | None) -> DecodedTokenPayload | None:
    return _decode_token(token, expected_type=REFRESH_TOKEN_TYPE)

