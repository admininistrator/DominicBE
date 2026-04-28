from alembic import op
import sqlalchemy as sa

revision = "0001_baseline_existing_chat_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "chat_sessions" not in existing_tables:
        op.create_table(
            "chat_sessions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("username", sa.String(length=255), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False, server_default="New chat"),
            sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_chat_sessions_id", "chat_sessions", ["id"])
        op.create_index("ix_chat_sessions_username", "chat_sessions", ["username"])

    if "users" not in existing_tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("username", sa.String(length=255), nullable=False),
            sa.Column("password", sa.String(length=255), nullable=False),
            sa.Column("max_tokens_per_day", sa.Integer(), nullable=True, server_default="10000"),
            sa.Column("total_token_used", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("total_input_tokens_used", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("total_output_tokens_used", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("last_token_reset_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("username", name="uq_users_username"),
        )
        op.create_index("ix_users_id", "users", ["id"])

    if "messages" not in existing_tables:
        op.create_table(
            "messages",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("session_id", sa.Integer(), nullable=True),
            sa.Column("request_id", sa.String(length=36), nullable=False),
            sa.Column("sender_username", sa.String(length=255), nullable=False),
            sa.Column("role", sa.Enum("user", "assistant", name="message_role"), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("input_tokens", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("output_tokens", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("status", sa.Enum("pending", "success", "error", name="message_status"), nullable=False, server_default="pending"),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        )
        op.create_index("ix_messages_id", "messages", ["id"])
        op.create_index("ix_messages_session_id", "messages", ["session_id"])
        op.create_index("ix_messages_request_id", "messages", ["request_id"])
        op.create_index("ix_messages_sender_username", "messages", ["sender_username"])

    if "chat_summaries" not in existing_tables:
        op.create_table(
            "chat_summaries",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("username", sa.String(length=255), nullable=False),
            sa.Column("session_id", sa.Integer(), nullable=False),
            sa.Column("summary_text", sa.Text(), nullable=False, server_default=""),
            sa.Column("last_summarized_message_id", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_chat_summaries_id", "chat_summaries", ["id"])
        op.create_index("ix_chat_summaries_username", "chat_summaries", ["username"])
        op.create_index("ix_chat_summaries_session_id", "chat_summaries", ["session_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "chat_summaries" in existing_tables:
        op.drop_index("ix_chat_summaries_session_id", table_name="chat_summaries")
        op.drop_index("ix_chat_summaries_username", table_name="chat_summaries")
        op.drop_index("ix_chat_summaries_id", table_name="chat_summaries")
        op.drop_table("chat_summaries")

    if "messages" in existing_tables:
        op.drop_index("ix_messages_sender_username", table_name="messages")
        op.drop_index("ix_messages_request_id", table_name="messages")
        op.drop_index("ix_messages_session_id", table_name="messages")
        op.drop_index("ix_messages_id", table_name="messages")
        op.drop_table("messages")

    if "users" in existing_tables:
        op.drop_index("ix_users_id", table_name="users")
        op.drop_table("users")

    if "chat_sessions" in existing_tables:
        op.drop_index("ix_chat_sessions_username", table_name="chat_sessions")
        op.drop_index("ix_chat_sessions_id", table_name="chat_sessions")
        op.drop_table("chat_sessions")
