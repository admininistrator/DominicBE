import logging
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.json_utils import ensure_json_mapping
from app.crud import crud_chat
from app.crud import crud_knowledge
from app.services.retrieval_service import build_retrieval_metadata_contract, search_knowledge
from app.services import llm_provider
from app.services.llm_provider import LLMError
from app.services.rag_core_client import RagCoreClientError, get_rag_core_client, is_rag_core_api_mode
from app.services.tavily_service import TavilySearchError, search_web

# Local aliases for readability (resolved from settings at module load)
CONTEXT_WINDOW_SIZE = settings.context_window_size
ROLLING_WINDOW_HOURS = settings.rolling_window_hours
SUMMARY_MAX_TOKENS = settings.summary_max_tokens
SUMMARY_TRIGGER_MESSAGES = settings.summary_trigger_messages
TOKEN_ESTIMATE_CHARS_PER_TOKEN = settings.token_estimate_chars_per_token

logger = get_logger(__name__)

SOURCE_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")
AUTO_TITLE_PLACEHOLDER_PATTERN = re.compile(r"^(new chat|chat\s+\d+)$", re.IGNORECASE)
AUTO_SESSION_TITLE_MODEL = "gpt-5.4-mini"
AUTO_SESSION_TITLE_MAX_CHARS = 72
AUTO_SESSION_TITLE_MAX_TOKENS = 24
AUTO_SESSION_TITLE_MAX_WORDS = 8


class ProviderRequestError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail

    @classmethod
    def from_llm_error(cls, e: LLMError) -> "ProviderRequestError":
        return cls(e.status_code, e.detail)


@dataclass
class PreparedChatTurn:
    username: str
    session_id: int
    session: object
    user_message: str
    knowledge_document_id: int | None
    model: str | None
    reasoning_effort: str | None
    existing_message_count: int
    knowledge_base_active: bool
    request_id: str
    user_msg_id: int
    request_kwargs: dict
    retrieval_result: dict | None
    web_search_result: dict
    sources: list[dict]


def _format_exception_chain(exc: BaseException | None) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or repr(current)
        parts.append(f"{current.__class__.__name__}: {message}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def _get_model() -> str:
    """Return the resolved LiteLLM model string (for logging)."""
    return llm_provider.resolve_model()


def _load_message_images(payload: str | None) -> list[str]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("Failed to decode message image payload.", exc_info=True)
        return []
    if isinstance(data, dict):
        images = data.get("images") or []
        return [item for item in images if isinstance(item, str) and item]
    return [item for item in data if isinstance(item, str) and item]


def _load_message_documents(payload: str | None) -> list[dict]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("Failed to decode message document payload.", exc_info=True)
        return []
    if not isinstance(data, dict):
        return []

    documents = data.get("documents") or []
    normalized_documents: list[dict] = []
    for item in documents:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        normalized_documents.append(
            {
                "id": item.get("id"),
                "title": title,
                "session_id": item.get("session_id"),
            }
        )
    return normalized_documents


def _load_message_sources(payload: str | None) -> list[dict]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("Failed to decode message source payload.", exc_info=True)
        return []
    if not isinstance(data, dict):
        return []

    sources = data.get("sources") or []
    normalized_sources: list[dict] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        raw_score = item.get("score")
        normalized_sources.append(
            {
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "title": title,
                "source_type": (item.get("source_type") or "web").strip() or "web",
                "score": float(raw_score) if raw_score is not None else None,
                "snippet": (item.get("snippet") or "").strip(),
                "source_uri": item.get("source_uri"),
                "rank": item.get("rank"),
                "url": item.get("url"),
                "domain": item.get("domain"),
                "page_number": item.get("page_number"),
                "page_range": item.get("page_range"),
                "section_key": item.get("section_key"),
                "section_title": item.get("section_title"),
                "section_level": item.get("section_level"),
                "char_start": item.get("char_start"),
                "char_end": item.get("char_end"),
            }
        )
    return normalized_sources


def _load_message_assistant_meta(payload: str | None) -> dict | None:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("Failed to decode message assistant meta payload.", exc_info=True)
        return None
    if not isinstance(data, dict):
        return None

    assistant_meta = data.get("assistant_meta") or {}
    if not isinstance(assistant_meta, dict):
        return None

    model_name = (assistant_meta.get("model") or "").strip()
    reasoning_effort = (assistant_meta.get("reasoning_effort") or "").strip().lower() or None
    display_text = (assistant_meta.get("display_text") or "").strip()
    if not model_name and not display_text:
        return None

    return {
        "model": model_name or None,
        "reasoning_effort": reasoning_effort,
        "display_text": display_text or model_name,
    }


def _resolve_display_model_name(model_name: str | None) -> str:
    explicit = (model_name or "").strip()
    if explicit:
        if explicit.startswith("openai/gh/"):
            return explicit.split("openai/gh/", 1)[1]
        if explicit.startswith("openai/"):
            return explicit.split("openai/", 1)[1]
        if explicit.startswith("gh/"):
            return explicit.split("gh/", 1)[1]
        return explicit

    resolved = llm_provider.resolve_model(model_name)
    if resolved.startswith("openai/gh/"):
        return resolved.split("openai/gh/", 1)[1]
    if resolved.startswith("openai/"):
        return resolved.split("openai/", 1)[1]
    return resolved


def _build_assistant_meta(model_name: str | None, reasoning_effort: str | None) -> dict:
    display_model = _resolve_display_model_name(model_name)
    normalized_effort = (reasoning_effort or "").strip().lower() or None
    display_text = f"{display_model} {normalized_effort}" if normalized_effort else display_model
    return {
        "model": display_model,
        "reasoning_effort": normalized_effort,
        "display_text": display_text,
    }


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


def delete_session(db: Session, username: str, session_id: int) -> bool:
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")
    return crud_chat.delete_chat_session(db, username, session_id)


def rename_session(db: Session, username: str, session_id: int, title: str) -> dict:
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")

    renamed_session = crud_chat.rename_chat_session(db, username, session_id, title)
    if not renamed_session:
        raise ValueError("Session not found.")

    return {
        "id": renamed_session.id,
        "username": renamed_session.username,
        "title": renamed_session.title,
        "created_at": renamed_session.created_at,
        "updated_at": renamed_session.updated_at,
    }


def get_session_history(
    db: Session,
    username: str,
    session_id: int,
    *,
    skip: int = 0,
    limit: int | None = None,
    before_id: int | None = None,
):
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")
    rows, has_more = crud_chat.get_session_messages(
        db,
        username,
        session_id,
        skip=skip,
        limit=limit,
        before_id=before_id,
    )
    assistant_request_ids = [m.request_id for m in rows if m.role == "assistant" and m.request_id]
    citations_by_request = _build_citations_by_request(db, assistant_request_ids)
    retrieval_by_request = _build_retrieval_by_request(db, assistant_request_ids)
    items = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "images": _load_message_images(m.__dict__.get("image_payload_json")),
            "documents": _load_message_documents(m.__dict__.get("image_payload_json")),
            "assistant_meta": _load_message_assistant_meta(m.__dict__.get("image_payload_json")) if m.role == "assistant" else None,
            "input_tokens": int(m.input_tokens or 0),
            "output_tokens": int(m.output_tokens or 0),
            "created_at": m.created_at,
            "request_id": m.request_id,
            "sources": (
                citations_by_request.get(m.request_id, []) + _load_message_sources(m.__dict__.get("image_payload_json"))
                if m.role == "assistant"
                else []
            ),
            "retrieval": retrieval_by_request.get(m.request_id) if m.role == "assistant" else None,
        }
        for m in rows
    ]
    next_before_id = items[0]["id"] if has_more and items else None
    return {
        "items": items,
        "pagination": {
            "skip": skip,
            "limit": limit,
            "before_id": before_id,
            "returned": len(items),
            "has_more": has_more,
            "next_before_id": next_before_id,
        },
    }


def _build_citations_by_request(db: Session, request_ids: list[str]) -> dict[str, list[dict]]:
    rows = crud_knowledge.list_answer_citations_by_request_ids(db, request_ids)
    grouped: dict[str, list[dict]] = {}
    for citation, document, _chunk in rows:
        grouped.setdefault(citation.request_id, []).append(
            {
                "document_id": citation.document_id,
                "chunk_id": citation.chunk_id,
                "title": document.title,
                "source_type": "knowledge",
                "score": float(citation.score) if citation.score is not None else None,
                "snippet": citation.quoted_text or "",
                "source_uri": document.source_uri,
                "rank": citation.rank,
                "url": None,
                "domain": None,
            }
        )
    return grouped


def _build_retrieval_by_request(db: Session, request_ids: list[str]) -> dict[str, dict]:
    rows = crud_knowledge.list_retrieval_events_by_request_ids(db, request_ids)
    grouped: dict[str, dict] = {}
    for event in rows:
        metadata = ensure_json_mapping(event.metadata_json)
        if not event.request_id:
            continue
        grouped[event.request_id] = {
            "used": int(metadata.get("returned") or 0) > 0,
            "top_k": int(event.top_k or 0),
            "returned": int(metadata.get("returned") or 0),
            "retrieval_id": event.id,
            "latency_ms": int(event.latency_ms or 0),
            "document_id": metadata.get("document_id"),
            "strategy": metadata.get("strategy"),
            "original_query": metadata.get("original_query") or event.query_text,
            "rewritten_query": metadata.get("rewritten_query") or event.query_text,
            "query_expansions": metadata.get("query_expansions") or [],
            "fallback_used": bool(metadata.get("fallback_used")),
            "fallback_reason": metadata.get("fallback_reason"),
            "evidence_strength": metadata.get("evidence_strength"),
            "answer_policy": metadata.get("answer_policy"),
            "packed_count": int(metadata.get("packed_count") or 0),
            "packed_token_estimate": int(metadata.get("packed_token_estimate") or 0),
            "web_search_used": bool(metadata.get("web_search_used")),
            "web_results_count": int(metadata.get("web_results_count") or 0),
            "web_search_query": metadata.get("web_search_query"),
            "web_latency_ms": int(metadata.get("web_latency_ms") or 0),
            "rag_mode": metadata.get("rag_mode"),
            "retrieval_scope": metadata.get("retrieval_scope"),
            "selected_document_id": metadata.get("selected_document_id"),
            "session_id": metadata.get("session_id"),
            "section_key": metadata.get("section_key"),
            "section_confidence": metadata.get("section_confidence"),
            "vector_store_attempted": bool(metadata.get("vector_store_attempted")),
            "vector_store_failed": bool(metadata.get("vector_store_failed")),
            "vector_store_error_type": metadata.get("vector_store_error_type"),
        }
    return grouped


def _estimate_input_tokens(messages: list[dict], system_prompt: str | None = None) -> int:
    # Heuristic: ~4 chars/token for mixed Vietnamese/English text.
    total_chars = sum(len((m.get("content") or "")) for m in messages)
    if system_prompt:
        total_chars += len(system_prompt)
    return max(1, total_chars // TOKEN_ESTIMATE_CHARS_PER_TOKEN)


def _calculate_max_output_tokens(
    *,
    estimated_input_tokens: int,
    remaining_tokens: int,
    context_window: int,
    model_max_output_tokens: int,
) -> int:
    context_window_remaining = context_window - estimated_input_tokens
    return min(
        model_max_output_tokens,
        remaining_tokens - estimated_input_tokens,
        context_window_remaining,
    )


def _is_generated_session_title(title: str | None) -> bool:
    normalized = (title or "").strip()
    if not normalized:
        return True
    return bool(AUTO_TITLE_PLACEHOLDER_PATTERN.fullmatch(normalized))


def _truncate_session_title(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return ""

    words = normalized.split(" ")
    if len(words) > AUTO_SESSION_TITLE_MAX_WORDS:
        normalized = " ".join(words[:AUTO_SESSION_TITLE_MAX_WORDS]).rstrip(" .,:;|-")

    if len(normalized) > AUTO_SESSION_TITLE_MAX_CHARS:
        normalized = normalized[:AUTO_SESSION_TITLE_MAX_CHARS].rstrip()
        if " " in normalized:
            normalized = normalized.rsplit(" ", 1)[0].rstrip()

    return normalized.rstrip(" .,:;|-")


def _fallback_session_title(user_message: str, assistant_reply: str | None = None) -> str:
    fallback_candidates = [assistant_reply or "", user_message]
    cleanup_patterns = [
        r"^(hãy|vui lòng|xin hãy)\s+",
        r"^(giúp tôi|cho tôi|tôi muốn|tôi cần)\s+",
        r"^(please|can you|could you|help me)\s+",
    ]

    for candidate in fallback_candidates:
        normalized = re.sub(r"\s+", " ", (candidate or "").strip())
        if not normalized:
            continue

        first_chunk = re.split(r"[\n.!?;]+", normalized, maxsplit=1)[0].strip()
        for pattern in cleanup_patterns:
            first_chunk = re.sub(pattern, "", first_chunk, flags=re.IGNORECASE).strip()

        truncated = _truncate_session_title(first_chunk)
        if truncated and not _is_generated_session_title(truncated):
            return truncated

    return "New chat"


def _normalize_session_title_candidate(candidate: str | None, *, fallback: str, user_message: str | None = None) -> str:
    normalized = re.sub(r"\s+", " ", (candidate or "").replace("\n", " ")).strip()
    normalized = normalized.strip("'\"`").strip()
    normalized = re.sub(r"^title\s*:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.rstrip(" .,:;|-_")
    normalized_prompt = re.sub(r"\s+", " ", (user_message or "").strip()).lower()

    if not normalized or _is_generated_session_title(normalized):
        return fallback

    if normalized.lower() == normalized_prompt:
        return fallback

    if len(normalized.split()) > AUTO_SESSION_TITLE_MAX_WORDS:
        return fallback

    normalized = _truncate_session_title(normalized)

    return normalized or fallback


def _generate_session_title(user_message: str, assistant_reply: str | None = None) -> str:
    fallback_title = _fallback_session_title(user_message, assistant_reply)
    normalized_reply = re.sub(r"\s+", " ", (assistant_reply or "").strip())
    if len(normalized_reply) > 800:
        normalized_reply = normalized_reply[:800].rstrip()
    prompt = (
        "Create a concise title for a new chat session based on the first user prompt and the assistant's first reply. "
        "Return only the title, with no quotes, no markdown, and no explanation. "
        "Keep it under 7 words when possible and make it specific to the actual topic resolved in the exchange.\n\n"
        f"User message:\n{user_message.strip()}\n\n"
        f"Assistant reply:\n{normalized_reply or '(empty)'}"
    )
    try:
        result = llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            model=AUTO_SESSION_TITLE_MODEL,
            max_tokens=AUTO_SESSION_TITLE_MAX_TOKENS,
        )
        return _normalize_session_title_candidate(
            result.get("text"),
            fallback=fallback_title,
            user_message=user_message,
        )
    except Exception:
        logger.info("Session auto-title generation failed; using fallback title.", exc_info=True)
        return fallback_title


def _maybe_autotitle_session(
    db: Session,
    username: str,
    session,
    user_message: str,
    assistant_reply: str | None,
    existing_message_count: int,
) -> str | None:
    if existing_message_count != 0:
        return None
    if not _is_generated_session_title(getattr(session, "title", None)):
        return None

    next_title = _generate_session_title(user_message, assistant_reply)
    renamed_session = crud_chat.rename_chat_session(db, username, session.id, next_title)
    if not renamed_session:
        return None
    return renamed_session.title


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
        result = llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        updated_summary = result["text"].strip()
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


def _build_sources(results: list[dict]) -> list[dict]:
    return [
        {
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "title": row["title"],
            "source_type": "knowledge",
            "score": row.get("score"),
            "rerank_score": row.get("rerank_score"),
            "snippet": row.get("snippet") or "",
            "source_uri": row.get("source_uri"),
            "rank": index,
            "url": None,
            "domain": None,
            "page_number": row.get("page_number"),
            "page_range": row.get("page_range"),
            "section_key": row.get("section_key"),
            "section_title": row.get("section_title"),
            "section_level": row.get("section_level"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
        }
        for index, row in enumerate(results, start=1)
    ]


def _build_web_sources(results: list[dict], *, start_rank: int = 1) -> list[dict]:
    return [
        {
            "document_id": None,
            "chunk_id": None,
            "title": row.get("title") or row.get("url") or f"Web source {index}",
            "source_type": "web",
            "score": row.get("score"),
            "rerank_score": None,
            "snippet": row.get("snippet") or "",
            "source_uri": row.get("url"),
            "rank": start_rank + index - 1,
            "url": row.get("url"),
            "domain": row.get("domain"),
        }
        for index, row in enumerate(results, start=1)
    ]


def _build_evidence_context(knowledge_results: list[dict], web_results: list[dict] | None = None) -> str:
    blocks: list[str] = []
    source_index = 1

    for row in knowledge_results:
        labels: list[str] = []
        if row.get("section_title"):
            labels.append(f"section={row.get('section_title')}")
        if row.get("page_range"):
            labels.append(f"pages={row.get('page_range')}")
        elif row.get("page_number") is not None:
            labels.append(f"page={row.get('page_number')}")
        label_suffix = " " + " ".join(labels) if labels else ""
        blocks.append(
            "\n".join(
                [
                    f"[Source {source_index}] type=knowledge title={row['title']} document_id={row['document_id']} chunk_id={row['chunk_id']} score={float(row.get('score') or 0):.3f}{label_suffix}",
                    row.get("content") or row.get("snippet") or "",
                ]
            )
        )

        source_index += 1

    for row in web_results or []:
        blocks.append(
            "\n".join(
                [
                    f"[Source {source_index}] type=web title={row.get('title') or row.get('url') or 'Web result'} url={row.get('url') or ''} domain={row.get('domain') or ''} score={float(row.get('score') or 0):.3f}",
                    row.get("snippet") or "",
                ]
            )
        )
        source_index += 1

    return "\n\n".join(blocks)


def _pack_retrieval_results(results: list[dict]) -> tuple[list[dict], int]:
    if is_rag_core_api_mode():
        try:
            payload = get_rag_core_client().pack_context(
                results=results,
                max_context_chunks=settings.retrieval_max_context_chunks,
                max_context_tokens=settings.retrieval_max_context_tokens,
            )
            return payload.get("packed_results") or [], int(payload.get("packed_token_estimate") or 0)
        except RagCoreClientError:
            logger.warning("rag-core API context packing failed; using local packer")

    packed_results: list[dict] = []
    packed_token_estimate = 0

    for row in results:
        token_estimate = int(row.get("token_estimate") or max(1, len((row.get("content") or "")) // 4))
        if len(packed_results) >= settings.retrieval_max_context_chunks:
            break
        if packed_results and packed_token_estimate + token_estimate > settings.retrieval_max_context_tokens:
            continue
        if not packed_results and token_estimate > settings.retrieval_max_context_tokens:
            packed_results.append({**row, "token_estimate": token_estimate})
            packed_token_estimate += token_estimate
            break

        packed_results.append({**row, "token_estimate": token_estimate})
        packed_token_estimate += token_estimate

    return packed_results, packed_token_estimate


def _compose_system_prompt(
    summary_text: str | None,
    retrieval_result: dict | None,
    *,
    knowledge_document_id: int | None = None,
    web_search_result: dict | None = None,
    knowledge_base_active: bool = True,
) -> str:
    retrieved_results = (retrieval_result or {}).get("packed_results") or (retrieval_result or {}).get("results") or []
    web_results = (web_search_result or {}).get("results") or []
    evidence_strength = (retrieval_result or {}).get("evidence_strength") or "none"
    answer_policy = (retrieval_result or {}).get("answer_policy") or "grounded"
    fallback_used = bool((retrieval_result or {}).get("fallback_used"))
    sections = ["You are Dominic, a helpful assistant."]

    if knowledge_base_active:
        sections.extend(
            [
                "If knowledge-base sources are provided below, prioritize them for answers about uploaded documents.",
                "When using those sources, cite them inline as [Source 1], [Source 2], etc.",
                "Never fabricate sources or claim a document says something unless it is supported by the evidence block below.",
            ]
        )
    elif web_results:
        sections.append(
            "No knowledge-base documents are attached for this turn. Treat the Tavily web evidence below as the primary factual grounding source, cite it inline as [Source 1], [Source 2], etc., and do not invent unsupported facts."
        )
    else:
        sections.append(
            "No knowledge-base evidence is attached to this session. Answer normally using the user's prompt and conversation context."
        )

    if summary_text:
        sections.append(
            "Conversation memory summary (may omit small details). Use this as background context:\n"
            + summary_text
        )

    if retrieved_results or web_results:
        sections.append("Evidence for this turn:\n" + _build_evidence_context(retrieved_results, web_results))

    if web_results:
        sections.append(
            "Web search evidence may be used for recent or external facts that are not covered by the uploaded knowledge base. When you rely on those results, prefer the cited sources over unsupported claims."
        )

    if not knowledge_base_active:
        return "\n\n".join(section for section in sections if section)

    if evidence_strength == "grounded":
        sections.append(
            "The retrieved evidence is strong enough for grounded answers. Prefer the evidence above for factual claims tied to uploaded knowledge."
        )
    elif evidence_strength == "weak":
        sections.append(
            "The retrieved evidence is weak or partial. You may answer cautiously, but explicitly note uncertainty and avoid strong document-specific claims beyond the provided evidence."
        )
    elif fallback_used:
        sections.append(
            "The retrieved context is a fallback seed from the selected document, not a confident semantic match. Use it only as tentative context and clearly state if the document evidence is still insufficient."
        )
    else:
        sections.append(
            "No strong matching knowledge-base evidence was retrieved for this turn. Do not claim the answer is grounded in uploaded documents unless evidence is actually provided."
        )

    if knowledge_document_id is not None and settings.retrieval_strict_grounding_for_scoped_docs:
        sections.append(
            "Because the user selected a specific knowledge document, if the evidence remains weak or insufficient then say you do not have enough evidence from that document."
        )

    if answer_policy == "insufficient_evidence":
        sections.append(
            "Answer with a concise insufficiency statement. Do not make document-specific claims and do not imply certainty."
        )
    elif answer_policy == "cautious_general":
        sections.append(
            "You may provide a cautious high-level answer, but explicitly say the current knowledge-base evidence is insufficient for a confident grounded claim."
        )

    return "\n\n".join(section for section in sections if section)


def _determine_answer_policy(
    retrieval_result: dict | None,
    *,
    knowledge_document_id: int | None = None,
) -> str:
    evidence_strength = (retrieval_result or {}).get("evidence_strength") or "none"
    results = (retrieval_result or {}).get("packed_results") or (retrieval_result or {}).get("results") or []
    top_result = results[0] if results else {}
    top_confidence = max(
        float(top_result.get("score") or 0.0),
        float(top_result.get("rerank_score") or 0.0),
    )
    top_lexical = float(top_result.get("lexical_score") or 0.0)
    top_semantic = float(top_result.get("semantic_score") or 0.0)
    has_direct_lexical_support = top_lexical >= settings.retrieval_min_lexical_score
    has_strong_lexical_support = top_lexical >= settings.retrieval_low_confidence_score
    has_high_semantic_support = top_semantic >= settings.retrieval_low_confidence_score

    if evidence_strength == "grounded":
        if knowledge_document_id is None:
            if has_strong_lexical_support:
                return "grounded"
            if results:
                return "cautious_general"
            return "insufficient_evidence"
        return "grounded"

    if (
        knowledge_document_id is not None
        and results
        and top_confidence >= settings.retrieval_low_confidence_score
        and (has_direct_lexical_support or has_high_semantic_support)
    ):
        return "grounded"

    if evidence_strength == "weak":
        return "cautious_general"

    if knowledge_document_id is not None and settings.retrieval_strict_grounding_for_scoped_docs:
        return "insufficient_evidence"

    if evidence_strength == "fallback":
        return "cautious_general"

    return "insufficient_evidence"


def _build_insufficient_evidence_reply(knowledge_document_id: int | None = None) -> str:
    if knowledge_document_id is not None:
        return (
            "Tôi chưa có đủ bằng chứng từ tài liệu đã chọn để trả lời chắc chắn câu hỏi này. "
            "Vui lòng cung cấp thêm tài liệu liên quan hoặc đặt câu hỏi sát nội dung tài liệu hơn."
        )
    return (
        "Tôi chưa có đủ bằng chứng từ knowledge base hiện tại để trả lời chắc chắn câu hỏi này. "
        "Bạn có thể tải thêm tài liệu hoặc đặt lại câu hỏi cụ thể hơn."
    )


def _build_web_search_payload(web_search_result: dict | None, answer_policy: str | None) -> dict | None:
    results = (web_search_result or {}).get("results") or []
    web_used = bool((web_search_result or {}).get("used")) or bool(results)
    if not web_used:
        return None

    return {
        "used": False,
        "top_k": 0,
        "returned": 0,
        "retrieval_id": None,
        "latency_ms": 0,
        "document_id": None,
        "strategy": "web_search",
        "original_query": (web_search_result or {}).get("query"),
        "rewritten_query": None,
        "query_expansions": [],
        "fallback_used": False,
        "fallback_reason": None,
        "evidence_strength": "web" if results else "none",
        "answer_policy": answer_policy,
        "packed_count": 0,
        "packed_token_estimate": 0,
        "web_search_used": True,
        "web_results_count": len(results),
        "web_search_query": (web_search_result or {}).get("query"),
        "web_latency_ms": int((web_search_result or {}).get("latency_ms") or 0),
        **build_retrieval_metadata_contract(
            rag_mode="direct_chat",
            retrieval_scope="none",
        ),
    }


def _linkify_web_sources_in_reply(ai_content: str, sources: list[dict]) -> str:
    text = (ai_content or "").strip()
    if not text:
        return ai_content

    web_sources_by_rank: dict[int, dict] = {}
    for source in sources:
        if source.get("source_type") != "web" or not source.get("url"):
            continue
        rank = source.get("rank")
        if isinstance(rank, int):
            web_sources_by_rank[rank] = source

    if not web_sources_by_rank:
        return ai_content

    used_ranks: list[int] = []

    def replace_citation(match: re.Match[str]) -> str:
        rank = int(match.group(1))
        source = web_sources_by_rank.get(rank)
        if not source:
            return match.group(0)
        if rank not in used_ranks:
            used_ranks.append(rank)
        return f"[Source {rank}]({source['url']})"

    linked_text = SOURCE_CITATION_PATTERN.sub(replace_citation, text)

    ordered_web_sources = [
        web_sources_by_rank[rank]
        for rank in sorted(web_sources_by_rank)
        if rank in used_ranks
    ]
    if not ordered_web_sources:
        ordered_web_sources = [web_sources_by_rank[rank] for rank in sorted(web_sources_by_rank)]

    footer_lines = [
        "",
        "Nguồn web:",
        *[
            f"- [Source {source['rank']}: {source.get('title') or source.get('domain') or source['url']}]({source['url']})"
            for source in ordered_web_sources
        ],
    ]
    footer = "\n".join(footer_lines)

    if footer.strip() and footer.strip() not in linked_text:
        linked_text = linked_text.rstrip() + "\n\n" + footer.strip()

    return linked_text


def _apply_answer_guardrails(
    ai_content: str,
    retrieval_result: dict | None,
    sources: list[dict],
    *,
    knowledge_document_id: int | None = None,
    web_search_result: dict | None = None,
    knowledge_base_active: bool = True,
) -> tuple[str, list[dict], str]:
    answer_policy = _determine_answer_policy(
        retrieval_result,
        knowledge_document_id=knowledge_document_id,
    )
    web_results = (web_search_result or {}).get("results") or []

    if not knowledge_base_active:
        if web_results:
            return ai_content, sources, "web_grounded"
        return ai_content, sources, "general_chat"

    if web_results and knowledge_document_id is None:
        web_policy = "grounded" if answer_policy == "grounded" else "web_grounded"
        return ai_content, sources, web_policy

    if not settings.answer_guardrails_enabled:
        return ai_content, sources, answer_policy

    if answer_policy == "grounded":
        return ai_content, sources, answer_policy

    if answer_policy == "cautious_general":
        guarded_reply = (
            "Dựa trên bằng chứng hiện có, tôi chưa thể khẳng định chắc chắn từ knowledge base. "
            + ai_content.strip()
        )
        guarded_sources = sources if settings.answer_guardrails_allow_weak_citations else []
        return guarded_reply, guarded_sources, answer_policy

    return _build_insufficient_evidence_reply(knowledge_document_id), [], answer_policy


def _build_retrieval_payload(retrieval_result: dict | None) -> dict | None:
    if not retrieval_result:
        return None
    return {
        "used": int(retrieval_result.get("returned") or 0) > 0,
        "top_k": int(retrieval_result.get("top_k") or 0),
        "returned": int(retrieval_result.get("returned") or 0),
        "retrieval_id": retrieval_result.get("retrieval_id"),
        "request_id": retrieval_result.get("request_id"),
        "session_scope": retrieval_result.get("session_scope"),
        "latency_ms": int(retrieval_result.get("latency_ms") or 0),
        "document_id": retrieval_result.get("document_id"),
        "strategy": retrieval_result.get("strategy"),
        "original_query": retrieval_result.get("original_query"),
        "rewritten_query": retrieval_result.get("rewritten_query"),
        "query_expansions": retrieval_result.get("query_expansions") or [],
        "fallback_used": bool(retrieval_result.get("fallback_used")),
        "fallback_reason": retrieval_result.get("fallback_reason"),
        "evidence_strength": retrieval_result.get("evidence_strength"),
        "answer_policy": retrieval_result.get("answer_policy"),
        "packed_count": int(retrieval_result.get("packed_count") or 0),
        "packed_token_estimate": int(retrieval_result.get("packed_token_estimate") or 0),
        "web_search_used": bool(retrieval_result.get("web_search_used")),
        "web_results_count": int(retrieval_result.get("web_results_count") or 0),
        "web_search_query": retrieval_result.get("web_search_query"),
        "web_latency_ms": int(retrieval_result.get("web_latency_ms") or 0),
        "rag_mode": retrieval_result.get("rag_mode"),
        "retrieval_scope": retrieval_result.get("retrieval_scope"),
        "selected_document_id": retrieval_result.get("selected_document_id"),
        "session_id": retrieval_result.get("session_id"),
        "section_key": retrieval_result.get("section_key"),
        "section_confidence": retrieval_result.get("section_confidence"),
        "vector_store_attempted": bool(retrieval_result.get("vector_store_attempted")),
        "vector_store_failed": bool(retrieval_result.get("vector_store_failed")),
        "vector_store_error_type": retrieval_result.get("vector_store_error_type"),
    }


def _build_chat_metadata_retrieval_result(
    *,
    rag_mode: str,
    retrieval_scope: str,
    session_id: int | None = None,
    selected_document_id: int | None = None,
    fallback_reason: str | None = None,
) -> dict:
    return {
        "query": None,
        "top_k": 0,
        "returned": 0,
        "retrieval_id": None,
        "request_id": None,
        "session_scope": retrieval_scope if retrieval_scope in {"session", "global"} else "none",
        "latency_ms": 0,
        "document_id": selected_document_id,
        "strategy": rag_mode,
        "original_query": None,
        "rewritten_query": None,
        "query_expansions": [],
        "fallback_used": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "evidence_strength": "none",
        "answer_policy": None,
        "packed_count": 0,
        "packed_token_estimate": 0,
        "web_search_used": False,
        "web_results_count": 0,
        "web_search_query": None,
        "web_latency_ms": 0,
        "results": [],
        "packed_results": [],
        **build_retrieval_metadata_contract(
            rag_mode=rag_mode,
            retrieval_scope=retrieval_scope,
            selected_document_id=selected_document_id,
            session_id=session_id,
            fallback_reason=fallback_reason,
        ),
    }


def _build_hybrid_context(db: Session, username: str, session_id: int):
    summary_row = _refresh_summary_if_needed(db, username, session_id)
    recent = crud_chat.get_recent_user_history(db, username, session_id, CONTEXT_WINDOW_SIZE)

    # Only include successfully completed messages so that failed/pending
    # user messages don't create consecutive-same-role sequences that Anthropic
    # would reject with a 400/403.
    success_messages = [m for m in recent if getattr(m, "status", "success") == "success"]
    raw_messages = [{"role": m.role, "content": m.content} for m in success_messages]
    formatted_messages = _sanitize_messages_for_api(raw_messages)

    summary_text = summary_row.summary_text if summary_row and summary_row.summary_text else None
    return summary_text, formatted_messages


def _should_use_knowledge_base(
    *,
    knowledge_document_id: int | None,
    session_documents: list,
) -> bool:
    if knowledge_document_id is not None:
        return True
    return bool(session_documents)


def _prepare_chat_turn(
    db: Session,
    username: str,
    session_id: int,
    user_message: str,
    knowledge_document_id: int | None = None,
    use_web_search: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    images: list[str] | None = None,
    image_media_types: list[str] | None = None,
) -> PreparedChatTurn:
    user = crud_chat.get_user_by_username(db, username)
    if not user:
        raise ValueError(f"User '{username}' not found.")
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")
    existing_message_count = crud_chat.count_session_messages(db, username, session_id)
    if knowledge_document_id is not None:
        knowledge_document = crud_knowledge.get_document(db, knowledge_document_id)
        if not knowledge_document or knowledge_document.owner_username != username:
            raise ValueError("Knowledge document not found.")
        if knowledge_document.session_id is not None and knowledge_document.session_id != session_id:
            raise ValueError("Knowledge document does not belong to this chat session.")
        selected_document_indexed = knowledge_document.status == "indexed"
    else:
        selected_document_indexed = False

    model_selection = llm_provider.get_provider_registry().select_model(
        model,
        reasoning_effort=reasoning_effort,
    )

    session_documents = crud_knowledge.list_documents(
        db,
        username,
        session_id=session_id,
        session_id_filter="session",
    )
    indexed_session_documents = crud_knowledge.list_documents(
        db,
        username,
        session_id=session_id,
        session_id_filter="session",
        indexed_only=True,
    )
    session_document_attachments = [
        {
            "id": document.id,
            "title": document.title,
            "session_id": document.session_id,
        }
        for document in session_documents
    ]
    knowledge_base_active = _should_use_knowledge_base(
        knowledge_document_id=knowledge_document_id,
        session_documents=indexed_session_documents,
    )
    if knowledge_document_id is not None:
        knowledge_base_active = selected_document_indexed

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
        images=images or None,
        documents=session_document_attachments or None,
        status="pending",
    )

    summary_text, formatted_messages = _build_hybrid_context(db, username, session_id)
    retrieval_result = None
    packed_results: list[dict] = []
    packed_token_estimate = 0

    if knowledge_base_active:
        retrieval_result = search_knowledge(
            db=db,
            owner_username=username,
            query=user_message,
            top_k=settings.retrieval_top_k,
            document_id=knowledge_document_id,
            session_id=session_id,
            request_id=request_id,
        )
        packed_results, packed_token_estimate = _pack_retrieval_results(retrieval_result.get("results") or [])
        retrieval_result["packed_count"] = len(packed_results)
        retrieval_result["packed_token_estimate"] = packed_token_estimate
        retrieval_result["packed_results"] = packed_results
    elif knowledge_document_id is not None:
        retrieval_result = _build_chat_metadata_retrieval_result(
            rag_mode="indexing_pending",
            retrieval_scope="document",
            session_id=session_id,
            selected_document_id=knowledge_document_id,
            fallback_reason="no_indexed_document",
        )
    elif session_documents:
        retrieval_result = _build_chat_metadata_retrieval_result(
            rag_mode="indexing_pending",
            retrieval_scope="session",
            session_id=session_id,
            fallback_reason="no_indexed_session_documents",
        )
    else:
        retrieval_result = _build_chat_metadata_retrieval_result(
            rag_mode="direct_chat",
            retrieval_scope="none",
            session_id=None,
        )

    web_search_result = {"used": False, "query": None, "latency_ms": 0, "results": []}
    if use_web_search:
        web_search_result = search_web(user_message, max_results=settings.web_search_max_results)
    if retrieval_result is not None:
        retrieval_result["web_search_used"] = bool(web_search_result.get("used"))
        retrieval_result["web_results_count"] = len(web_search_result.get("results") or [])
        retrieval_result["web_search_query"] = web_search_result.get("query")
        retrieval_result["web_latency_ms"] = int(web_search_result.get("latency_ms") or 0)
        retrieval_result["answer_policy"] = _determine_answer_policy(
            retrieval_result,
            knowledge_document_id=knowledge_document_id,
        )

    if retrieval_result is not None and retrieval_result.get("retrieval_id"):
        crud_knowledge.update_retrieval_event_metadata(
            db,
            retrieval_result["retrieval_id"],
            {
                "packed_count": len(packed_results),
                "packed_token_estimate": packed_token_estimate,
                "answer_policy": retrieval_result["answer_policy"],
                "web_search_used": retrieval_result["web_search_used"],
                "web_results_count": retrieval_result["web_results_count"],
                "web_search_query": retrieval_result["web_search_query"],
                "web_latency_ms": retrieval_result["web_latency_ms"],
            },
        )

    knowledge_sources = _build_sources(packed_results)
    web_sources = _build_web_sources(
        web_search_result.get("results") or [],
        start_rank=len(knowledge_sources) + 1,
    )
    sources = [*knowledge_sources, *web_sources]
    system_prompt = _compose_system_prompt(
        summary_text,
        retrieval_result,
        knowledge_document_id=knowledge_document_id,
        web_search_result=web_search_result,
        knowledge_base_active=knowledge_base_active,
    )
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

    max_output_tokens = _calculate_max_output_tokens(
        estimated_input_tokens=estimated_input_tokens,
        remaining_tokens=remaining_tokens,
        context_window=model_selection.context_window,
        model_max_output_tokens=model_selection.max_output_tokens,
    )
    if max_output_tokens <= 0:
        crud_chat.update_message_tokens_and_status(
            db=db,
            message_id=user_msg.id,
            status="error",
            error_message="No remaining output token budget or context window capacity.",
        )
        raise PermissionError("Prompt vuot qua gioi han context window hoac quota token hien tai.")

    request_kwargs = {
        "messages": request_messages,
        "system": system_prompt or None,
        "max_tokens": max_output_tokens,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    if images:
        if not settings.llm_vision_enabled:
            raise ValueError(
                "Image attachments are not supported: vision is disabled on this server."
            )
        request_kwargs["images"] = images
        request_kwargs["image_media_types"] = image_media_types or []

    return PreparedChatTurn(
        username=username,
        session_id=session_id,
        session=session,
        user_message=user_message,
        knowledge_document_id=knowledge_document_id,
        model=model,
        reasoning_effort=reasoning_effort,
        existing_message_count=existing_message_count,
        knowledge_base_active=knowledge_base_active,
        request_id=request_id,
        user_msg_id=user_msg.id,
        request_kwargs=request_kwargs,
        retrieval_result=retrieval_result,
        web_search_result=web_search_result,
        sources=sources,
    )


def _finalize_chat_turn(
    db: Session,
    prepared: PreparedChatTurn,
    *,
    ai_content: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    sources = list(prepared.sources)
    retrieval_result = prepared.retrieval_result
    web_search_result = prepared.web_search_result

    ai_content, sources, answer_policy = _apply_answer_guardrails(
        ai_content,
        retrieval_result,
        sources,
        knowledge_document_id=prepared.knowledge_document_id,
        web_search_result=web_search_result,
        knowledge_base_active=prepared.knowledge_base_active,
    )
    autotitle_source_reply = ai_content
    ai_content = _linkify_web_sources_in_reply(ai_content, sources)
    if retrieval_result is not None:
        retrieval_result["answer_policy"] = answer_policy

    if retrieval_result is not None and retrieval_result.get("retrieval_id"):
        crud_knowledge.update_retrieval_event_metadata(
            db,
            retrieval_result["retrieval_id"],
            {"answer_policy": answer_policy},
        )

    crud_chat.update_message_tokens_and_status(
        db=db,
        message_id=prepared.user_msg_id,
        input_tokens=input_tokens,
        status="success",
        error_message=None,
    )

    persisted_web_sources = [source for source in sources if source.get("source_type") == "web"]
    assistant_meta = _build_assistant_meta(prepared.model, prepared.reasoning_effort)

    crud_chat.create_message(
        db=db,
        role="assistant",
        sender_username=prepared.username,
        session_id=prepared.session_id,
        content=ai_content,
        request_id=prepared.request_id,
        sources=persisted_web_sources or None,
        assistant_meta=assistant_meta,
        input_tokens=0,
        output_tokens=output_tokens,
        status="success",
    )

    _maybe_autotitle_session(
        db,
        prepared.username,
        prepared.session,
        prepared.user_message,
        autotitle_source_reply,
        prepared.existing_message_count,
    )

    crud_knowledge.replace_answer_citations(
        db,
        prepared.request_id,
        citations=[
            {
                "document_id": source["document_id"],
                "chunk_id": source["chunk_id"],
                "rank": source["rank"],
                "score": source.get("score"),
                "quoted_text": source.get("snippet") or "",
            }
            for source in sources
            if source.get("source_type") == "knowledge"
        ],
    )

    crud_chat.touch_chat_session(db, prepared.session_id)
    crud_chat.increment_user_tokens(db, prepared.username, input_tokens, output_tokens)

    return {
        "reply": ai_content,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "request_id": prepared.request_id,
        "sources": sources,
        "assistant_meta": assistant_meta,
        "retrieval": _build_retrieval_payload(retrieval_result)
        or _build_web_search_payload(web_search_result, answer_policy),
    }


def _mark_chat_turn_failed(db: Session, prepared: PreparedChatTurn | None, detail: str):
    if not prepared:
        return
    crud_chat.update_message_tokens_and_status(
        db=db,
        message_id=prepared.user_msg_id,
        status="error",
        error_message=detail,
    )


def handle_chat(
    db: Session,
    username: str,
    session_id: int,
    user_message: str,
    knowledge_document_id: int | None = None,
    use_web_search: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    images: list[str] | None = None,
    image_media_types: list[str] | None = None,
):
    prepared: PreparedChatTurn | None = None

    try:
        prepared = _prepare_chat_turn(
            db,
            username,
            session_id,
            user_message,
            knowledge_document_id=knowledge_document_id,
            use_web_search=use_web_search,
            model=model,
            reasoning_effort=reasoning_effort,
            images=images,
            image_media_types=image_media_types,
        )

        logger.info(
            "LiteLLM call username=%s session_id=%s model=%s messages=%d images=%d",
            username, session_id, llm_provider.resolve_model(model), len(prepared.request_kwargs["messages"]), len(images or []),
        )
        llm_result = llm_provider.complete(**prepared.request_kwargs)
        return _finalize_chat_turn(
            db,
            prepared,
            ai_content=llm_result["text"],
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
        )

    except LLMError as e:
        logger.warning(
            "LLM call failed username=%s session_id=%s model=%s: %s",
            username, session_id, llm_provider.resolve_model(model), e.detail,
            exc_info=True,
        )
        _mark_chat_turn_failed(db, prepared, e.detail)
        raise ProviderRequestError.from_llm_error(e) from e

    except TavilySearchError as e:
        logger.warning(
            "Tavily search failed username=%s session_id=%s: %s",
            username, session_id, e.detail,
            exc_info=True,
        )
        _mark_chat_turn_failed(db, prepared, e.detail)
        raise ProviderRequestError(e.status_code, e.detail) from e

    except Exception as e:
        _mark_chat_turn_failed(db, prepared, str(e))
        raise


def handle_chat_stream(
    db: Session,
    username: str,
    session_id: int,
    user_message: str,
    knowledge_document_id: int | None = None,
    use_web_search: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    images: list[str] | None = None,
    image_media_types: list[str] | None = None,
) -> Iterator[dict]:
    prepared: PreparedChatTurn | None = None
    try:
        prepared = _prepare_chat_turn(
            db,
            username,
            session_id,
            user_message,
            knowledge_document_id=knowledge_document_id,
            use_web_search=use_web_search,
            model=model,
            reasoning_effort=reasoning_effort,
            images=images,
            image_media_types=image_media_types,
        )

        yield {"event": "start", "data": {"request_id": prepared.request_id}}

        stream_result: dict | None = None
        for chunk in llm_provider.stream_complete(**prepared.request_kwargs):
            chunk_type = chunk.get("type")
            if chunk_type == "delta":
                delta_text = chunk.get("text") or ""
                if delta_text:
                    yield {
                        "event": "delta",
                        "data": {"text": delta_text, "request_id": prepared.request_id},
                    }
            elif chunk_type == "complete":
                stream_result = chunk

        if stream_result is None:
            raise RuntimeError("Streaming provider completed without a final payload.")

        final_result = _finalize_chat_turn(
            db,
            prepared,
            ai_content=stream_result["text"],
            input_tokens=stream_result["input_tokens"],
            output_tokens=stream_result["output_tokens"],
        )
        yield {"event": "final", "data": {"success": True, **final_result}}

    except LLMError as e:
        logger.warning(
            "LLM stream failed username=%s session_id=%s model=%s: %s",
            username, session_id, llm_provider.resolve_model(model), e.detail,
            exc_info=True,
        )
        _mark_chat_turn_failed(db, prepared, e.detail)
        raise ProviderRequestError.from_llm_error(e) from e
    except TavilySearchError as e:
        logger.warning(
            "Tavily search failed during stream username=%s session_id=%s: %s",
            username, session_id, e.detail,
            exc_info=True,
        )
        _mark_chat_turn_failed(db, prepared, e.detail)
        raise ProviderRequestError(e.status_code, e.detail) from e
    except Exception as e:
        _mark_chat_turn_failed(db, prepared, str(e))
        raise
