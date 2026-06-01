"""Add celery_task_id to ingestion_jobs

Revision ID: 0010_add_celery_task_id
Revises: 0009_auth_token_version
"""
from alembic import op
import sqlalchemy as sa


revision = "0010_add_celery_task_id"
down_revision = "0009_auth_token_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "celery_task_id")
