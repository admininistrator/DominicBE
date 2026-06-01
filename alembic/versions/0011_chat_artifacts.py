"""Add first-class chat artifacts table

Revision ID: 0011_chat_artifacts
Revises: 0010_add_celery_task_id
"""
from alembic import op
import sqlalchemy as sa


revision = "0011_chat_artifacts"
down_revision = "0010_add_celery_task_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content_json", sa.Text(), nullable=True),
        sa.Column("elements_json", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifacts_id", "artifacts", ["id"], unique=False)
    op.create_index("ix_artifacts_session_id", "artifacts", ["session_id"], unique=False)
    op.create_index("ix_artifacts_message_id", "artifacts", ["message_id"], unique=False)
    op.create_index("ix_artifacts_kind", "artifacts", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_artifacts_kind", table_name="artifacts")
    op.drop_index("ix_artifacts_message_id", table_name="artifacts")
    op.drop_index("ix_artifacts_session_id", table_name="artifacts")
    op.drop_index("ix_artifacts_id", table_name="artifacts")
    op.drop_table("artifacts")
