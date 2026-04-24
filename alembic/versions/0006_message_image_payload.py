"""Persist chat message image attachments

Revision ID: 0006_message_image_payload
Revises: 0005_drop_plaintext_password
"""
# noinspection PyUnresolvedReferences
from alembic import op
import sqlalchemy as sa

revision = "0006_message_image_payload"
down_revision = "0005_drop_plaintext_password"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("image_payload_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "image_payload_json")


