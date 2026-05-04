from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.database import SessionLocal
from app.core.logging import set_log_context
from app.core.security import decode_access_token
from app.crud import crud_auth


bearer_scheme = HTTPBearer(auto_error=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db=Depends(get_db),
):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_payload = decode_access_token(credentials.credentials)
    if not token_payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = crud_auth.get_user_by_username(db, token_payload["username"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user does not exist.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    current_version = int(getattr(user, "auth_token_version", 0) or 0)
    if current_version != token_payload["token_version"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.auth_username = user.username
    set_log_context(username=user.username)
    return user


def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db=Depends(get_db),
):
    if not credentials or credentials.scheme.lower() != "bearer":
        return None

    token_payload = decode_access_token(credentials.credentials)
    if not token_payload:
        return None

    user = crud_auth.get_user_by_username(db, token_payload["username"])
    if not user:
        return None

    current_version = int(getattr(user, "auth_token_version", 0) or 0)
    if current_version != token_payload["token_version"]:
        return None

    request.state.auth_username = user.username
    set_log_context(username=user.username)
    return user


def require_admin(current_user=Depends(get_current_user)):
    """Dependency that ensures the current user has admin role."""
    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return current_user
