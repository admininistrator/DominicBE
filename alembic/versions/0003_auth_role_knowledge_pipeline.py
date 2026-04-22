"""add user role, reset token, and upload_size to config

Revision ID: 0003_auth_role_knowledge_pipeline
Revises: 0002_foundation_hardening_knowledge
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_auth_role_knowledge_pipeline"
down_revision = "0002_foundation_hardening_knowledge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add role column to users
    op.add_column("users", sa.Column("role", sa.String(length=50), nullable=False, server_default="user"))
    # Add reset token columns
    op.add_column("users", sa.Column("reset_token", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("reset_token_expires_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "reset_token_expires_at")
    op.drop_column("users", "reset_token")
    op.drop_column("users", "role")

