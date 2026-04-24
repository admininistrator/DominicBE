from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    role = Column(String(50), nullable=False, server_default="user")  # "user" | "admin"
    reset_token = Column(String(255), nullable=True)
    reset_token_expires_at = Column(DateTime, nullable=True)
    max_tokens_per_day = Column(Integer, default=10000)
    total_token_used = Column(Integer, default=0)
    total_input_tokens_used = Column(Integer, default=0)
    total_output_tokens_used = Column(Integer, default=0)
    last_token_reset_at = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    request_id = Column(String(36), nullable=False, index=True)
    sender_username = Column(String(255), nullable=False, index=True)
    role = Column(Enum("user", "assistant"), nullable=False)
    content = Column(Text, nullable=False)
    image_payload_json = Column(Text, nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    status = Column(Enum("pending", "success", "error"), nullable=False, server_default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ChatSummary(Base):
    __tablename__ = "chat_summaries"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(255), nullable=False, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    summary_text = Column(Text, nullable=False, default="")
    last_summarized_message_id = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(255), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New chat")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

