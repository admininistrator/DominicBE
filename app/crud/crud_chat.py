import json

from sqlalchemy import inspect
from sqlalchemy.orm import Session, load_only
from secrets import compare_digest
from app.models.chat_models import Message, User, ChatSummary, ChatSession
from app.core.security import (
    hash_password,
    normalize_password,
    normalize_username,
    password_hash_needs_update,
    verify_password,
)
from uuid import uuid4
from datetime import datetime
from datetime import timedelta
from sqlalchemy import func


_MESSAGE_IMAGE_PAYLOAD_SUPPORT: dict[str, bool] = {}


def _bind_cache_key(db: Session) -> str:
    bind = db.get_bind()
    return str(getattr(getattr(bind, "engine", bind), "url", "default"))


def message_image_payload_supported(db: Session) -> bool:
    cache_key = _bind_cache_key(db)
    if cache_key not in _MESSAGE_IMAGE_PAYLOAD_SUPPORT:
        inspector = inspect(db.get_bind())
        columns = {column["name"] for column in inspector.get_columns("messages")}
        _MESSAGE_IMAGE_PAYLOAD_SUPPORT[cache_key] = "image_payload_json" in columns
    return _MESSAGE_IMAGE_PAYLOAD_SUPPORT[cache_key]


def _message_query(db: Session):
    attrs = [
        Message.id,
        Message.session_id,
        Message.request_id,
        Message.sender_username,
        Message.role,
        Message.content,
        Message.input_tokens,
        Message.output_tokens,
        Message.status,
        Message.error_message,
        Message.created_at,
    ]
    if message_image_payload_supported(db):
        attrs.append(Message.image_payload_json)
    return db.query(Message).options(load_only(*attrs))

def create_message(
    db: Session,
    role: str,
    sender_username: str,
    session_id: int,
    content: str,
    request_id: str | None,
    images: list[str] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    status: str = "pending",
    error_message: str | None = None,
):
    if not request_id:
        request_id = str(uuid4())
    values = {
        "session_id": session_id,
        "request_id": request_id,
        "role": role,
        "sender_username": sender_username,
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": status,
        "error_message": error_message,
    }
    if images and message_image_payload_supported(db):
        values["image_payload_json"] = json.dumps(images)

    insert_result = db.execute(Message.__table__.insert().values(**values))
    db.commit()
    message_id = insert_result.inserted_primary_key[0]
    return _message_query(db).filter(Message.id == message_id).first()


def get_user_history(db: Session, username: str):
    return (
        _message_query(db)
        .filter(Message.sender_username == username)
        .order_by(Message.created_at.asc())
        .all()
    )


def create_chat_session(db: Session, username: str, title: str | None = None):
    row = ChatSession(
        username=username,
        title=(title or "New chat").strip() or "New chat",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_chat_session(db: Session, username: str, session_id: int):
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.username == username)
        .first()
    )


def list_chat_sessions(db: Session, username: str):
    return (
        db.query(ChatSession)
        .filter(ChatSession.username == username)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .all()
    )


def delete_chat_session(db: Session, username: str, session_id: int) -> bool:
    row = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.username == username).first()
    if not row:
        return False
    # Delete all messages in this session first
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.delete(row)
    db.commit()
    return True


def rename_chat_session(db: Session, username: str, session_id: int, title: str) -> ChatSession | None:
    row = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.username == username).first()
    if not row:
        return None
    row.title = title.strip() or row.title
    db.commit()
    db.refresh(row)
    return row


def touch_chat_session(db: Session, session_id: int):
    row = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if row:
        row.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(row)
    return row


def get_session_messages(db: Session, username: str, session_id: int):
    return (
        _message_query(db)
        .filter(
            Message.sender_username == username,
            Message.session_id == session_id,
        )
        .order_by(Message.id.asc())
        .all()
    )


def get_recent_user_history(db: Session, username: str, session_id: int, limit: int):
    rows = (
        _message_query(db)
        .filter(
            Message.sender_username == username,
            Message.session_id == session_id,
        )
        .order_by(Message.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def get_messages_for_summary(db: Session, username: str, session_id: int, after_id: int, before_id: int):
    return (
        _message_query(db)
        .filter(
            Message.sender_username == username,
            Message.session_id == session_id,
            Message.id > after_id,
            Message.id < before_id,
        )
        .order_by(Message.id.asc())
        .all()
    )


def get_chat_summary(db: Session, username: str, session_id: int):
    return (
        db.query(ChatSummary)
        .filter(ChatSummary.username == username, ChatSummary.session_id == session_id)
        .first()
    )


def upsert_chat_summary(db: Session, username: str, session_id: int, summary_text: str, last_message_id: int):
    row = (
        db.query(ChatSummary)
        .filter(ChatSummary.username == username, ChatSummary.session_id == session_id)
        .first()
    )
    if not row:
        row = ChatSummary(
            username=username,
            session_id=session_id,
            summary_text=summary_text,
            last_summarized_message_id=last_message_id,
        )
        db.add(row)
    else:
        row.summary_text = summary_text
        row.last_summarized_message_id = last_message_id
    db.commit()
    db.refresh(row)
    return row


def get_user_by_username(db: Session, username: str):
    normalized_username = normalize_username(username)
    if not normalized_username:
        return None
    return db.query(User).filter(User.username == normalized_username).first()


def create_user(db: Session, username: str, password: str, max_tokens_per_day: int = 10000):
    normalized_username = normalize_username(username)
    if not normalized_username:
        raise ValueError("Username must not be empty.")

    existing = get_user_by_username(db, normalized_username)
    if existing:
        raise ValueError("Username already exists.")

    user = User(
        username=normalized_username,
        password=None,
        password_hash=hash_password(password, enforce_policy=True),
        max_tokens_per_day=max_tokens_per_day,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def reset_user_tokens_if_needed(db: Session, user: User, reset_interval_hours: int = 2):
    now = datetime.utcnow()
    last_reset = user.last_token_reset_at or user.created_at or now
    if now - last_reset >= timedelta(hours=reset_interval_hours):
        user.total_token_used = 0
        user.total_input_tokens_used = 0
        user.total_output_tokens_used = 0
        user.last_token_reset_at = now
        db.commit()
        db.refresh(user)
    return user


def get_rolling_token_usage(db: Session, username: str, window_hours: int = 2):
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    totals = (
        db.query(
            func.coalesce(func.sum(Message.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(Message.output_tokens), 0).label("output_tokens"),
        )
        .filter(
            Message.sender_username == username,
            Message.created_at >= cutoff,
            Message.status == "success",
        )
        .first()
    )
    input_tokens = int(totals.input_tokens or 0)
    output_tokens = int(totals.output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def verify_user_credentials(db: Session, username: str, password: str):
    normalized_username = normalize_username(username)
    if not normalized_username:
        return None

    user = db.query(User).filter(User.username == normalized_username).first()
    if not user:
        return None

    normalized_password = normalize_password(password)
    if not normalized_password:
        return None

    stored_hash = (user.password_hash or "").strip()
    legacy_password = normalize_password(user.password)

    if stored_hash and verify_password(normalized_password, stored_hash):
        if password_hash_needs_update(stored_hash) or user.password:
            user.password_hash = hash_password(normalized_password)
            user.password = None
            db.commit()
            db.refresh(user)
        return user

    if legacy_password and compare_digest(legacy_password, normalized_password):
        user.password_hash = hash_password(normalized_password)
        user.password = None
        db.commit()
        db.refresh(user)
        return user

    return None

def get_user_usage(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    return {
        "username": user.username,
        "max_tokens_per_day": int(user.max_tokens_per_day or 10000),
        "total_token_used": int(user.total_token_used or 0),
        "total_input_tokens_used": int(user.total_input_tokens_used or 0),
        "total_output_tokens_used": int(user.total_output_tokens_used or 0),
    }


def increment_user_tokens(db: Session, username: str, input_tokens: int, output_tokens: int):
    user = db.query(User).filter(User.username == username).first()
    if user:
        user = reset_user_tokens_if_needed(db, user, reset_interval_hours=2)
        if user.total_token_used is None:
            user.total_token_used = 0
        if getattr(user, 'total_input_tokens_used', None) is None:
            user.total_input_tokens_used = 0
        if getattr(user, 'total_output_tokens_used', None) is None:
            user.total_output_tokens_used = 0

        user.total_input_tokens_used += input_tokens
        user.total_output_tokens_used += output_tokens
        user.total_token_used += (input_tokens + output_tokens)
        db.commit()
        db.refresh(user)
    return user


def update_message_tokens(db: Session, message_id: int, input_tokens: int = None, output_tokens: int = None):
    return update_message_tokens_and_status(
        db=db,
        message_id=message_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def update_message_status(db: Session, message_id: int, status: str, error_message: str | None = None):
    return update_message_tokens_and_status(
        db=db,
        message_id=message_id,
        status=status,
        error_message=error_message,
    )


def update_message_tokens_and_status(
    db: Session,
    message_id: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    status: str | None = None,
    error_message: str | None = None,
):
    msg = _message_query(db).filter(Message.id == message_id).first()
    if not msg:
        return None

    if input_tokens is not None:
        msg.input_tokens = input_tokens
    if output_tokens is not None:
        msg.output_tokens = output_tokens
    if status is not None:
        msg.status = status
    msg.error_message = error_message

    db.commit()
    db.refresh(msg)
    return msg
