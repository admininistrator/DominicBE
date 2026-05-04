"""Schemas for auth endpoints."""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.security import validate_username_policy


class _UsernameInputModel(BaseModel):
    @field_validator("username", check_fields=False)
    @classmethod
    def _validate_username(cls, value: str) -> str:
        return validate_username_policy(value, field_name="Username", min_length=1, max_length=255)


class LoginRequest(_UsernameInputModel):
    username: str
    password: str


class RegisterRequest(_UsernameInputModel):
    username: str = Field(min_length=3, max_length=255)
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class LoginResponse(BaseModel):
    success: bool
    username: str
    role: str = "user"
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class MeResponse(BaseModel):
    username: str
    role: str = "user"


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str
    confirm_new_password: str

    @model_validator(mode="after")
    def passwords_match(self):
        if self.new_password != self.confirm_new_password:
            raise ValueError("New passwords do not match.")
        return self


class ResetPasswordRequest(_UsernameInputModel):
    """Admin issues a reset token for a target user."""
    username: str
    expire_minutes: int = Field(default=30, ge=5, le=1440)


class ResetPasswordResponse(BaseModel):
    username: str
    reset_token: str
    expire_minutes: int


class ConsumeResetTokenRequest(_UsernameInputModel):
    username: str
    reset_token: str
    new_password: str
    confirm_new_password: str

    @model_validator(mode="after")
    def passwords_match(self):
        if self.new_password != self.confirm_new_password:
            raise ValueError("New passwords do not match.")
        return self


class SetRoleRequest(_UsernameInputModel):
    username: str
    role: str = Field(pattern="^(user|admin)$")


class UserSummary(BaseModel):
    id: int
    username: str
    role: str
    max_tokens_per_day: int
    created_at: datetime


class MessageResponse(BaseModel):
    success: bool
    message: str

