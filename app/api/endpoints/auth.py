"""Auth router – register, login, password management, admin user ops."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_admin
from app.models.chat_models import User
from app.schemas.auth_schemas import (
    ChangePasswordRequest,
    ConsumeResetTokenRequest,
    LoginRequest,
    LoginResponse,
    MeResponse,
    MessageResponse,
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
    register_user,
    set_user_role,
)

router = APIRouter()


# ---- Public ----------------------------------------------------------------

@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        result = register_user(db, request.username, request.password)
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        result = login_user(db, request.username, request.password)
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)):
    result = get_me(current_user)
    return MeResponse(**result)


# ---- Authenticated user ----------------------------------------------------

@router.post("/change-password", response_model=MessageResponse)
def change_pwd(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = change_password(db, current_user, request.old_password, request.new_password)
        return MessageResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-password", response_model=MessageResponse)
def reset_pwd(request: ConsumeResetTokenRequest, db: Session = Depends(get_db)):
    """Public endpoint: user provides reset token + new password."""
    try:
        result = consume_reset_token(db, request.username, request.reset_token, request.new_password)
        return MessageResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Admin only ------------------------------------------------------------

@router.post("/admin/reset-password", response_model=ResetPasswordResponse)
def admin_reset_pwd(
    request: ResetPasswordRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin generates a reset token for any user."""
    try:
        result = admin_reset_password(db, request.username, request.expire_minutes)
        return ResetPasswordResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/set-role", response_model=MessageResponse)
def admin_set_role(
    request: SetRoleRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    try:
        result = set_user_role(db, request.username, request.role)
        return MessageResponse(success=True, message=f"User '{result['username']}' role set to '{result['role']}'.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/users", response_model=list[UserSummary])
def admin_list_users(
    skip: int = 0,
    limit: int = 100,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_users(db, skip, limit)

