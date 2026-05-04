"""Add shared rate limit store table

Revision ID: 0008_shared_rate_limit_store
Revises: 0007_knowledge_session_id
"""

from alembic import op
import sqlalchemy as sa

revision = "0008_shared_rate_limit_store"
down_revision = "0007_knowledge_session_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_buckets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("scope", sa.String(length=128), nullable=False),
        sa.Column("bucket_key", sa.String(length=255), nullable=False),
        sa.Column("window_start_epoch", sa.BigInteger(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scope",
            "bucket_key",
            "window_start_epoch",
            name="uq_rate_limit_buckets_scope_key_window",
        ),
    )
    op.create_index("ix_rate_limit_buckets_id", "rate_limit_buckets", ["id"], unique=False)
    op.create_index("ix_rate_limit_buckets_scope", "rate_limit_buckets", ["scope"], unique=False)
    op.create_index("ix_rate_limit_buckets_bucket_key", "rate_limit_buckets", ["bucket_key"], unique=False)
    op.create_index(
        "ix_rate_limit_buckets_window_start_epoch",
        "rate_limit_buckets",
        ["window_start_epoch"],
        unique=False,
    )
    op.create_index("ix_rate_limit_buckets_updated_at", "rate_limit_buckets", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_rate_limit_buckets_updated_at", table_name="rate_limit_buckets")
    op.drop_index("ix_rate_limit_buckets_window_start_epoch", table_name="rate_limit_buckets")
    op.drop_index("ix_rate_limit_buckets_bucket_key", table_name="rate_limit_buckets")
    op.drop_index("ix_rate_limit_buckets_scope", table_name="rate_limit_buckets")
    op.drop_index("ix_rate_limit_buckets_id", table_name="rate_limit_buckets")
    op.drop_table("rate_limit_buckets")