"""Add session_id to knowledge_documents for per-chat knowledge

Revision ID: 0007_knowledge_session_id
Revises: 0006_message_image_payload
"""
# noinspection PyUnresolvedReferences
from alembic import op
import sqlalchemy as sa

revision = "0007_knowledge_session_id"
down_revision = "0006_message_image_payload"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_documents",
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_documents", "session_id")
