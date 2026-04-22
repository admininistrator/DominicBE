"""Phase 6 – production readiness: soft delete + audit_logs table

Revision ID: 0004_phase6_production_readiness
Revises: 0003_auth_role_knowledge_pipeline
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_phase6_production_readiness"
down_revision = "0003_auth_role_knowledge_pipeline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Soft-delete column on knowledge_documents
    op.add_column(
        "knowledge_documents",
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_knowledge_documents_deleted_at",
        "knowledge_documents",
        ["deleted_at"],
        unique=False,
    )

    # 2. Audit logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("actor_username", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("request_id", sa.String(length=36), nullable=True),
        sa.Column("detail_json", sa.JSON(), nullable=True),
        sa.Column("result_code", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_id", "audit_logs", ["id"], unique=False)
    op.create_index("ix_audit_logs_actor_username", "audit_logs", ["actor_username"], unique=False)
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index("ix_audit_logs_resource_type", "audit_logs", ["resource_type"], unique=False)
    op.create_index("ix_audit_logs_resource_id", "audit_logs", ["resource_id"], unique=False)
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"], unique=False)
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_request_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_resource_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_resource_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_username", table_name="audit_logs")
    op.drop_index("ix_audit_logs_id", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_knowledge_documents_deleted_at", table_name="knowledge_documents")
    op.drop_column("knowledge_documents", "deleted_at")

