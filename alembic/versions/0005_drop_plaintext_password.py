"""Phase 6.1 – drop plaintext password column from users table

Revision ID: 0005_drop_plaintext_password
Revises: 0004_phase6_production_readiness
"""
from alembic import op
import sqlalchemy as sa

revision = "0005_drop_plaintext_password"
down_revision = "0004_phase6_production_readiness"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove the legacy plaintext password column.
    # All authentication now uses password_hash (bcrypt).
    op.drop_column("users", "password")


def downgrade() -> None:
    # Re-add the column as nullable so existing rows are not broken.
    op.add_column(
        "users",
        sa.Column("password", sa.String(length=255), nullable=True),
    )

