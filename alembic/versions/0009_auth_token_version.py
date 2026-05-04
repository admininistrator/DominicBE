"""Add auth token version to users

Revision ID: 0009_auth_token_version
Revises: 0008_shared_rate_limit_store
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_auth_token_version"
down_revision = "0008_shared_rate_limit_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("auth_token_version", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("users", "auth_token_version")