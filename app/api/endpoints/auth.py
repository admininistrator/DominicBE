"""Auth router – register, login, password management, admin user ops."""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_admin
from app.api.error_handling import raise_internal_server_error
from app.core.config import settings
from app.core.logging import get_logger, set_log_context
from app.core.rate_limit import get_client_ip
from app.crud import crud_knowledge
from app.models.chat_models import User
from app.schemas.auth_schemas import (
    ChangePasswordRequest,
    ConsumeResetTokenRequest,
    LoginRequest,
    LoginResponse,
    MeResponse,
    MessageResponse,
    RefreshTokenRequest,
    RegisterRequest,
    ResetPasswordRequest,
    ResetPasswordResponse,
    SetRoleRequest,
    UserSummary,
)
from app.services.auth_service import (
    admin_reset_password,
    change_password,
    consume_reset_token,
    get_me,
    list_users,
    login_user,
    logout_user,
    refresh_auth_tokens,
    register_user,
    set_user_role,
)

router = APIRouter()
logger = get_logger(__name__)


def _auth_request_detail(request: Request, **kwargs) -> dict:
    return {
        "client_ip": get_client_ip(request, trust_proxy_headers=settings.rate_limit_trust_proxy_headers),
        "method": request.method,
        "path": request.url.path,
        **kwargs,
    }


def _audit_auth_action(
    db: Session,
    *,
    actor_username: str,
    action: str,
    request: Request,
    resource_id: str | None = None,
    result_code: int | None = None,
    detail_json: dict | None = None,
):
    if not settings.audit_log_enabled:
        return
    try:
        crud_knowledge.create_audit_log(
            db,
            actor_username,
            action,
            resource_type="user",
            resource_id=resource_id or actor_username,
            request_id=getattr(request.state, "request_id", None),
            detail_json=_auth_request_detail(request, **(detail_json or {})),
            result_code=result_code,
        )
    except Exception:
        logger.warning("auth audit write failed action=%s actor=%s", action, actor_username, exc_info=True)


# ---- Public ----------------------------------------------------------------

@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, http_request: Request, db: Session = Depends(get_db)):
    set_log_context(username=request.username)
    try:
        result = register_user(db, request.username, request.password)
        _audit_auth_action(
            db,
            actor_username=result["username"],
            action="auth.register",
            request=http_request,
            result_code=status.HTTP_201_CREATED,
            detail_json={"role": result["role"]},
        )
        logger.info("auth register succeeded status=%s", status.HTTP_201_CREATED)
        return LoginResponse(**result)
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.register",
            request=http_request,
            result_code=status.HTTP_400_BAD_REQUEST,
            detail_json={"error": str(e)},
        )
        logger.warning("auth register failed status=%s detail=%s", status.HTTP_400_BAD_REQUEST, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.register",
            request=http_request,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.register", exc=e)


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, http_request: Request, db: Session = Depends(get_db)):
    set_log_context(username=request.username)
    try:
        result = login_user(db, request.username, request.password)
        _audit_auth_action(
            db,
            actor_username=result["username"],
            action="auth.login",
            request=http_request,
            result_code=status.HTTP_200_OK,
            detail_json={"role": result["role"]},
        )
        logger.info("auth login succeeded status=%s", status.HTTP_200_OK)
        return LoginResponse(**result)
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.login",
            request=http_request,
            result_code=status.HTTP_401_UNAUTHORIZED,
            detail_json={"error": str(e)},
        )
        logger.warning("auth login failed status=%s detail=%s", status.HTTP_401_UNAUTHORIZED, e)
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.login",
            request=http_request,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.login", exc=e)


@router.post("/refresh", response_model=LoginResponse)
def refresh(request: RefreshTokenRequest, http_request: Request, db: Session = Depends(get_db)):
    try:
        result = refresh_auth_tokens(db, request.refresh_token)
        set_log_context(username=result["username"])
        _audit_auth_action(
            db,
            actor_username=result["username"],
            action="auth.refresh",
            request=http_request,
            result_code=status.HTTP_200_OK,
            detail_json={"role": result["role"]},
        )
        logger.info("auth refresh succeeded status=%s", status.HTTP_200_OK)
        return LoginResponse(**result)
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username="-",
            action="auth.refresh",
            request=http_request,
            result_code=status.HTTP_401_UNAUTHORIZED,
            detail_json={"error": str(e)},
        )
        logger.warning("auth refresh failed status=%s detail=%s", status.HTTP_401_UNAUTHORIZED, e)
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username="-",
            action="auth.refresh",
            request=http_request,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.refresh", exc=e)


@router.get("/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)):
    result = get_me(current_user)
    return MeResponse(**result)


# ---- Authenticated user ----------------------------------------------------

@router.post("/change-password", response_model=MessageResponse)
def change_pwd(
    request: ChangePasswordRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    set_log_context(username=current_user.username)
    try:
        result = change_password(db, current_user, request.old_password, request.new_password)
        _audit_auth_action(
            db,
            actor_username=current_user.username,
            action="auth.change_password",
            request=http_request,
            result_code=status.HTTP_200_OK,
        )
        logger.info("auth change_password succeeded status=%s", status.HTTP_200_OK)
        return MessageResponse(**result)
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username=current_user.username,
            action="auth.change_password",
            request=http_request,
            result_code=status.HTTP_400_BAD_REQUEST,
            detail_json={"error": str(e)},
        )
        logger.warning("auth change_password failed status=%s detail=%s", status.HTTP_400_BAD_REQUEST, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=current_user.username,
            action="auth.change_password",
            request=http_request,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.change_password", exc=e)


@router.post("/logout", response_model=MessageResponse)
def logout(
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    set_log_context(username=current_user.username)
    try:
        result = logout_user(db, current_user)
        _audit_auth_action(
            db,
            actor_username=current_user.username,
            action="auth.logout",
            request=http_request,
            result_code=status.HTTP_200_OK,
        )
        logger.info("auth logout succeeded status=%s", status.HTTP_200_OK)
        return MessageResponse(**result)
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=current_user.username,
            action="auth.logout",
            request=http_request,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.logout", exc=e)


@router.post("/reset-password", response_model=MessageResponse)
def reset_pwd(request: ConsumeResetTokenRequest, http_request: Request, db: Session = Depends(get_db)):
    """Public endpoint: user provides reset token + new password."""
    set_log_context(username=request.username)
    try:
        result = consume_reset_token(db, request.username, request.reset_token, request.new_password)
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.reset_password",
            request=http_request,
            result_code=status.HTTP_200_OK,
        )
        logger.info("auth reset_password succeeded status=%s", status.HTTP_200_OK)
        return MessageResponse(**result)
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.reset_password",
            request=http_request,
            result_code=status.HTTP_400_BAD_REQUEST,
            detail_json={"error": str(e)},
        )
        logger.warning("auth reset_password failed status=%s detail=%s", status.HTTP_400_BAD_REQUEST, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=request.username,
            action="auth.reset_password",
            request=http_request,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.reset_password", exc=e)


# ---- Admin only ------------------------------------------------------------

@router.post("/admin/reset-password", response_model=ResetPasswordResponse)
def admin_reset_pwd(
    request: ResetPasswordRequest,
    http_request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin generates a reset token for any user."""
    set_log_context(username=admin.username)
    try:
        result = admin_reset_password(db, request.username, request.expire_minutes)
        _audit_auth_action(
            db,
            actor_username=admin.username,
            action="auth.admin_reset_password",
            request=http_request,
            resource_id=request.username,
            result_code=status.HTTP_200_OK,
            detail_json={"target_username": request.username, "expire_minutes": request.expire_minutes},
        )
        logger.info("auth admin_reset_password succeeded status=%s target=%s", status.HTTP_200_OK, request.username)
        return ResetPasswordResponse(**result)
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username=admin.username,
            action="auth.admin_reset_password",
            request=http_request,
            resource_id=request.username,
            result_code=status.HTTP_404_NOT_FOUND,
            detail_json={"target_username": request.username, "error": str(e)},
        )
        logger.warning("auth admin_reset_password failed status=%s detail=%s", status.HTTP_404_NOT_FOUND, e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=admin.username,
            action="auth.admin_reset_password",
            request=http_request,
            resource_id=request.username,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"target_username": request.username, "error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.admin_reset_password", exc=e)


@router.post("/admin/set-role", response_model=MessageResponse)
def admin_set_role(
    request: SetRoleRequest,
    http_request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    set_log_context(username=admin.username)
    try:
        result = set_user_role(db, request.username, request.role)
        _audit_auth_action(
            db,
            actor_username=admin.username,
            action="auth.admin_set_role",
            request=http_request,
            resource_id=request.username,
            result_code=status.HTTP_200_OK,
            detail_json={"target_username": request.username, "role": request.role},
        )
        logger.info("auth admin_set_role succeeded status=%s target=%s role=%s", status.HTTP_200_OK, request.username, request.role)
        return MessageResponse(success=True, message=f"User '{result['username']}' role set to '{result['role']}'.")
    except ValueError as e:
        _audit_auth_action(
            db,
            actor_username=admin.username,
            action="auth.admin_set_role",
            request=http_request,
            resource_id=request.username,
            result_code=status.HTTP_400_BAD_REQUEST,
            detail_json={"target_username": request.username, "role": request.role, "error": str(e)},
        )
        logger.warning("auth admin_set_role failed status=%s detail=%s", status.HTTP_400_BAD_REQUEST, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _audit_auth_action(
            db,
            actor_username=admin.username,
            action="auth.admin_set_role",
            request=http_request,
            resource_id=request.username,
            result_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail_json={"target_username": request.username, "role": request.role, "error": type(e).__name__},
        )
        raise_internal_server_error(logger, action="auth.admin_set_role", exc=e)


@router.get("/admin/users", response_model=list[UserSummary])
def admin_list_users(
    http_request: Request,
    skip: int = 0,
    limit: int = 100,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    set_log_context(username=admin.username)
    users = list_users(db, skip, limit)
    _audit_auth_action(
        db,
        actor_username=admin.username,
        action="auth.admin_list_users",
        request=http_request,
        result_code=status.HTTP_200_OK,
        detail_json={"skip": skip, "limit": limit, "returned": len(users)},
    )
    logger.info("auth admin_list_users succeeded status=%s returned=%s", status.HTTP_200_OK, len(users))
    return users

