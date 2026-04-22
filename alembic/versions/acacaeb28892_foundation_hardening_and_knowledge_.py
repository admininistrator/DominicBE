from alembic import op
import sqlalchemy as sa

revision = "0002_foundation_hardening_knowledge"
down_revision = "0001_baseline_existing_chat_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(length=255), nullable=True))
    op.alter_column(
        "users",
        "password",
        existing_type=sa.String(length=255),
        nullable=True,
    )

    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_username", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column(
            "source_type",
            sa.Enum("upload", "text", "url", name="knowledge_source_type"),
            nullable=False,
            server_default="text",
        ),
        sa.Column("source_uri", sa.String(length=1024), nullable=True),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.Enum("uploaded", "processing", "indexed", "failed", name="knowledge_document_status"),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_knowledge_documents_owner_username", "knowledge_documents", ["owner_username"])
    op.create_index("ix_knowledge_documents_status", "knowledge_documents", ["status"])
    op.create_index("ix_knowledge_documents_checksum", "knowledge_documents", ["checksum"])

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("vector_id", sa.String(length=255), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_knowledge_chunks_document_id", "knowledge_chunks", ["document_id"])
    op.create_index("ix_knowledge_chunks_vector_id", "knowledge_chunks", ["vector_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("queued", "processing", "completed", "failed", name="ingestion_job_status"),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])

    op.create_table(
        "retrieval_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=36), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_retrieval_events_username", "retrieval_events", ["username"])
    op.create_index("ix_retrieval_events_session_id", "retrieval_events", ["session_id"])
    op.create_index("ix_retrieval_events_request_id", "retrieval_events", ["request_id"])

    op.create_table(
        "answer_citations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("quoted_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["knowledge_chunks.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_answer_citations_request_id", "answer_citations", ["request_id"])
    op.create_index("ix_answer_citations_document_id", "answer_citations", ["document_id"])
    op.create_index("ix_answer_citations_chunk_id", "answer_citations", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_answer_citations_chunk_id", table_name="answer_citations")
    op.drop_index("ix_answer_citations_document_id", table_name="answer_citations")
    op.drop_index("ix_answer_citations_request_id", table_name="answer_citations")
    op.drop_table("answer_citations")

    op.drop_index("ix_retrieval_events_request_id", table_name="retrieval_events")
    op.drop_index("ix_retrieval_events_session_id", table_name="retrieval_events")
    op.drop_index("ix_retrieval_events_username", table_name="retrieval_events")
    op.drop_table("retrieval_events")

    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_document_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    op.drop_index("ix_knowledge_chunks_vector_id", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_document_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")

    op.drop_index("ix_knowledge_documents_checksum", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_status", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_owner_username", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")

    op.drop_column("users", "password_hash")
    op.alter_column(
        "users",
        "password",
        existing_type=sa.String(length=255),
        nullable=False,
    )
