import logging
import os
from uuid import uuid4

from anthropic import APIConnectionError, APIStatusError, Anthropic
import httpx
from sqlalchemy.orm import Session

from app.core.config import (
    CONTEXT_WINDOW_SIZE,
    MAX_OUTPUT_TOKENS,
    ROLLING_WINDOW_HOURS,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TRIGGER_MESSAGES,
    TOKEN_ESTIMATE_CHARS_PER_TOKEN,
)
from app.crud import crud_chat

logger = logging.getLogger("uvicorn.error")


class ProviderRequestError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# Lazy-initialised so that a missing API key does NOT crash the app at import time.
# Also re-initialised when the API key env-var changes (e.g. after updating Azure App Settings).
_client: Anthropic | None = None
_client_key: str | None = None  # the key used to build the current _client
_client_base_url: str | None = None
_client_force_ipv4: bool | None = None


def _get_base_url() -> str | None:
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    return base_url or None


def _should_force_ipv4() -> bool:
    raw = (os.getenv("ANTHROPIC_FORCE_IPV4") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_client() -> Anthropic:
    global _client, _client_key, _client_base_url, _client_force_ipv4
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    base_url = _get_base_url()
    force_ipv4 = _should_force_ipv4()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please configure it in Azure App Service → Configuration → Application settings."
        )
    if (
        _client is None
        or api_key != _client_key
        or base_url != _client_base_url
        or force_ipv4 != _client_force_ipv4
    ):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if force_ipv4:
            client_kwargs["http_client"] = httpx.Client(
                transport=httpx.HTTPTransport(local_address="0.0.0.0")
            )
        _client = Anthropic(**client_kwargs)
        _client_key = api_key
        _client_base_url = base_url
        _client_force_ipv4 = force_ipv4
        logger.info(
            "Anthropic client (re)initialised. key prefix=%s model=%s base_url=%s force_ipv4=%s",
            api_key[:12] + "...",
            _get_model(),
            base_url or "(default)",
            force_ipv4,
        )
    return _client


def _get_model() -> str:
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    if not model:
        model = "claude-3-5-haiku-latest"
    return model.strip()


def _extract_provider_message(exc: APIStatusError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return str(exc).strip()


def _raise_provider_error(exc: Exception):
    model = _get_model()

    if isinstance(exc, APIConnectionError):
        raise ProviderRequestError(
            503,
            "Khong the ket noi toi nha cung cap AI (Anthropic). Kiem tra outbound network, DNS, firewall, va API key.",
        ) from exc

    if isinstance(exc, APIStatusError):
        status_code = int(getattr(exc, "status_code", 502) or 502)
        provider_message = _extract_provider_message(exc)

        if status_code == 401:
            detail = "Anthropic API key khong hop le hoac chua duoc cap quyen. Kiem tra bien moi truong ANTHROPIC_API_KEY tren Azure App Service."
        elif status_code == 403:
            detail = (
                f"Anthropic tu choi request cho model '{model}'. "
                "Neu key/model nay goi duoc tren may local nhung fail tren Azure App Service, kha nang cao la Anthropic dang chan theo region/egress IP cua Azure (vi du East Asia) hoac request phai di qua proxy/base URL khac. "
                "Kiem tra billing/quyen truy cap model, thu model khac, hoac cau hinh ANTHROPIC_BASE_URL neu ban co AI gateway/proxy. "
                f"Provider message: {provider_message}"
            )
        elif status_code == 404:
            detail = f"Model '{model}' khong ton tai hoac API key hien tai khong duoc phep dung model nay."
        elif status_code == 429:
            detail = "Anthropic dang gioi han toc do hoac het quota. Thu lai sau hoac kiem tra usage/billing."
        else:
            detail = f"Anthropic API loi {status_code}: {provider_message}"

        raise ProviderRequestError(status_code, detail) from exc

    raise exc

def login_user(db: Session, username: str, password: str):
    user = crud_chat.verify_user_credentials(db, username, password)
    if not user:
        raise ValueError("Invalid username or password.")
    return {"success": True, "username": user.username}


def get_usage(db: Session, username: str):
    user = crud_chat.get_user_by_username(db, username)
    if not user:
        raise ValueError(f"User '{username}' not found.")
    usage = crud_chat.get_user_usage(db, username)
    rolling = crud_chat.get_rolling_token_usage(db, username, window_hours=ROLLING_WINDOW_HOURS)
    return {
        "username": usage["username"],
        "max_tokens_per_day": usage["max_tokens_per_day"],
        # Compatibility keys now explicitly represent rolling window.
        "total_token_used": rolling["total_tokens"],
        "total_input_tokens_used": rolling["input_tokens"],
        "total_output_tokens_used": rolling["output_tokens"],
        # Explicit lifetime counters from users table.
        "lifetime_total_token_used": usage["total_token_used"],
        "lifetime_total_input_tokens_used": usage["total_input_tokens_used"],
        "lifetime_total_output_tokens_used": usage["total_output_tokens_used"],
        # Explicit rolling counters used for quota enforcement.
        "rolling_window_hours": ROLLING_WINDOW_HOURS,
        "rolling_total_token_used": rolling["total_tokens"],
        "rolling_input_tokens_used": rolling["input_tokens"],
        "rolling_output_tokens_used": rolling["output_tokens"],
    }


def create_session(db: Session, username: str, title: str | None = None):
    user = crud_chat.get_user_by_username(db, username)
    if not user:
        raise ValueError(f"User '{username}' not found.")
    row = crud_chat.create_chat_session(db, username, title)
    return {
        "id": row.id,
        "username": row.username,
        "title": row.title,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def get_sessions(db: Session, username: str):
    user = crud_chat.get_user_by_username(db, username)
    if not user:
        raise ValueError(f"User '{username}' not found.")
    rows = crud_chat.list_chat_sessions(db, username)
    return [
        {
            "id": r.id,
            "username": r.username,
            "title": r.title,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        }
        for r in rows
    ]


def get_session_history(db: Session, username: str, session_id: int):
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")
    rows = crud_chat.get_session_messages(db, username, session_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "input_tokens": int(m.input_tokens or 0),
            "output_tokens": int(m.output_tokens or 0),
            "created_at": m.created_at,
        }
        for m in rows
    ]


def _estimate_input_tokens(messages: list[dict], system_prompt: str | None = None) -> int:
    # Prefer provider-side token counting when available.
    try:
        kwargs: dict = {"model": _get_model(), "messages": messages}
        if system_prompt:
            kwargs["system"] = system_prompt
        count_tokens = getattr(_get_client().messages, "count_tokens", None)
        if not callable(count_tokens):
            raise AttributeError("Anthropic SDK does not expose messages.count_tokens")
        token_info = count_tokens(**kwargs)
        return int(getattr(token_info, "input_tokens", 0) or 0)
    except Exception:
        # Fallback heuristic: ~4 chars/token for mixed VN/EN text.
        total_chars = sum(len((m.get("content") or "")) for m in messages)
        if system_prompt:
            total_chars += len(system_prompt)
        return max(1, total_chars // TOKEN_ESTIMATE_CHARS_PER_TOKEN)


def _build_summary_prompt(old_summary: str, messages: list) -> str:
    transcript = "\n".join([f"[{m.role}] {m.content}" for m in messages])
    return (
        "You are a memory compressor for chat context. "
        "Create a concise factual summary that preserves user preferences, goals, constraints, and unresolved tasks. "
        "Do not include fluff.\n\n"
        f"Previous summary:\n{old_summary or '(none)'}\n\n"
        f"New transcript chunk:\n{transcript}\n\n"
        "Return updated summary only."
    )


def _refresh_summary_if_needed(db: Session, username: str, session_id: int):
    summary_row = crud_chat.get_chat_summary(db, username, session_id)
    last_done_id = int(summary_row.last_summarized_message_id or 0) if summary_row else 0

    recent = crud_chat.get_recent_user_history(db, username, session_id, CONTEXT_WINDOW_SIZE)
    if not recent:
        return summary_row

    recent_first_id = recent[0].id
    candidates = crud_chat.get_messages_for_summary(
        db=db,
        username=username,
        session_id=session_id,
        after_id=last_done_id,
        before_id=recent_first_id,
    )

    if len(candidates) < SUMMARY_TRIGGER_MESSAGES:
        return summary_row

    try:
        prompt = _build_summary_prompt(summary_row.summary_text if summary_row else "", candidates)
        resp = _get_client().messages.create(
            model=_get_model(),
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        updated_summary = resp.content[0].text.strip()
        return crud_chat.upsert_chat_summary(
            db=db,
            username=username,
            session_id=session_id,
            summary_text=updated_summary,
            last_message_id=candidates[-1].id,
        )
    except Exception:
        # If summarization fails, continue with recent window only.
        return summary_row


def _sanitize_messages_for_api(messages: list[dict]) -> list[dict]:
    """Ensure messages alternate user/assistant and start with user.

    Anthropic requires strictly alternating roles.  When a previous request
    failed the user message was persisted but no assistant reply was saved,
    leaving two consecutive user messages.  We merge/drop them so the payload
    is always valid.
    """
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if not role or not msg.get("content"):
            continue
        if result and result[-1]["role"] == role:
            # Consecutive same-role: merge into the existing message.
            result[-1] = {
                "role": role,
                "content": result[-1]["content"] + "\n\n" + msg["content"],
            }
        else:
            result.append({"role": role, "content": msg["content"]})

    # Must start with a user message (drop leading assistant messages).
    while result and result[0]["role"] != "user":
        result.pop(0)

    return result


def _build_request_messages(history_messages: list[dict], user_message: str) -> list[dict]:
    """Build a valid Anthropic payload that always includes the latest user prompt."""
    latest_user_message = (user_message or "").strip()
    combined = list(history_messages)
    if latest_user_message:
        combined.append({"role": "user", "content": latest_user_message})

    sanitized = _sanitize_messages_for_api(combined)
    if sanitized and sanitized[-1]["role"] != "user" and latest_user_message:
        sanitized.append({"role": "user", "content": latest_user_message})

    return sanitized


def _build_hybrid_context(db: Session, username: str, session_id: int):
    summary_row = _refresh_summary_if_needed(db, username, session_id)
    recent = crud_chat.get_recent_user_history(db, username, session_id, CONTEXT_WINDOW_SIZE)

    # Only include successfully completed messages so that failed/pending
    # user messages don't create consecutive-same-role sequences that Anthropic
    # would reject with a 400/403.
    success_messages = [m for m in recent if getattr(m, "status", "success") == "success"]
    raw_messages = [{"role": m.role, "content": m.content} for m in success_messages]
    formatted_messages = _sanitize_messages_for_api(raw_messages)

    system_prompt = None
    if summary_row and summary_row.summary_text:
        system_prompt = (
            "Conversation memory summary (may omit small details). "
            "Use this as background context:\n" + summary_row.summary_text
        )
    return system_prompt, formatted_messages


def handle_chat(db: Session, username: str, session_id: int, user_message: str):
    user = crud_chat.get_user_by_username(db, username)
    if not user:
        raise ValueError(f"User '{username}' not found.")
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")

    daily_limit = int(user.max_tokens_per_day or 10000)
    rolling = crud_chat.get_rolling_token_usage(db, username, window_hours=ROLLING_WINDOW_HOURS)
    total_used = int(rolling["total_tokens"])
    remaining_tokens = daily_limit - total_used
    if remaining_tokens <= 0:
        raise PermissionError(
            f"Rolling {ROLLING_WINDOW_HOURS}-hour token limit exceeded ({total_used}/{daily_limit})."
        )

    request_id = str(uuid4())

    user_msg = crud_chat.create_message(
        db=db,
        role="user",
        sender_username=username,
        session_id=session_id,
        content=user_message,
        request_id=request_id,
        status="pending",
    )

    try:
        system_prompt, formatted_messages = _build_hybrid_context(db, username, session_id)
        request_messages = _build_request_messages(formatted_messages, user_message)

        if not request_messages:
            crud_chat.update_message_tokens_and_status(
                db=db,
                message_id=user_msg.id,
                status="error",
                error_message="Current prompt is empty after sanitization.",
            )
            raise ValueError("Prompt khong hop le hoac dang rong.")

        estimated_input_tokens = _estimate_input_tokens(request_messages, system_prompt)
        if estimated_input_tokens >= remaining_tokens:
            crud_chat.update_message_tokens_and_status(
                db=db,
                message_id=user_msg.id,
                status="error",
                error_message=f"Rolling {ROLLING_WINDOW_HOURS}-hour token limit exceeded before API call.",
            )
            raise PermissionError(f"Rolling {ROLLING_WINDOW_HOURS}-hour token limit exceeded.")

        max_output_tokens = min(MAX_OUTPUT_TOKENS, remaining_tokens - estimated_input_tokens)
        if max_output_tokens <= 0:
            crud_chat.update_message_tokens_and_status(
                db=db,
                message_id=user_msg.id,
                status="error",
                error_message="No remaining output token budget.",
            )
            raise PermissionError(f"Rolling {ROLLING_WINDOW_HOURS}-hour token limit exceeded.")

        request_kwargs = {
            "model": _get_model(),
            "max_tokens": max_output_tokens,
            "messages": request_messages,
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt
        logger.info(
            "Sending Anthropic request username=%s session_id=%s model=%s messages=%s first_role=%s last_role=%s",
            username,
            session_id,
            request_kwargs["model"],
            len(request_messages),
            request_messages[0]["role"] if request_messages else None,
            request_messages[-1]["role"] if request_messages else None,
        )
        response = _get_client().messages.create(**request_kwargs)

        ai_content = response.content[0].text
        in_tokens = int(response.usage.input_tokens or 0)
        out_tokens = int(response.usage.output_tokens or 0)

        crud_chat.update_message_tokens_and_status(
            db=db,
            message_id=user_msg.id,
            input_tokens=in_tokens,
            status="success",
            error_message=None,
        )

        crud_chat.create_message(
            db=db,
            role="assistant",
            sender_username=username,
            session_id=session_id,
            content=ai_content,
            request_id=request_id,
            input_tokens=0,
            output_tokens=out_tokens,
            status="success",
        )

        crud_chat.touch_chat_session(db, session_id)

        crud_chat.increment_user_tokens(db, username, in_tokens, out_tokens)

        return {
            "reply": ai_content,
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
            "request_id": request_id,
        }

    except (APIConnectionError, APIStatusError) as e:
        logger.warning("Anthropic request failed for username=%s session_id=%s: %s", username, session_id, e)
        provider_error = None
        try:
            _raise_provider_error(e)
        except ProviderRequestError as mapped_error:
            provider_error = mapped_error

        crud_chat.update_message_tokens_and_status(
            db=db,
            message_id=user_msg.id,
            status="error",
            error_message=provider_error.detail if provider_error else str(e),
        )
        raise provider_error if provider_error else e

    except Exception as e:
        crud_chat.update_message_tokens_and_status(
            db=db,
            message_id=user_msg.id,
            status="error",
            error_message=str(e),
        )
        raise
