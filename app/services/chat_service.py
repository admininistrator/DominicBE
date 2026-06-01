import logging
import asyncio
import concurrent.futures
import json
import re
import unicodedata
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
from app.services.mcp.adapters import normalize_tool_result
from app.services.artifacts.diagram_intent import is_diagram_intent
from app.services.artifacts.excalidraw_schema import (
    ExcalidrawValidationError,
    build_excalidraw_scene,
    normalize_excalidraw_elements,
)
from app.services.artifacts.excalidraw_streaming import (
    artifact_delta_event,
    artifact_done_event,
    artifact_error_event,
    artifact_id_for_request,
    artifact_response_from_elements,
    artifact_start_event,
    build_diagram_system_prompt,
    parse_final_elements,
    partial_json_array_objects,
    repair_final_elements,
)
from app.services.artifacts.generation import (
    ArtifactGenerationRequest,
    ArtifactGenerationResult,
    ArtifactGenerationService,
    CallableArtifactProvider,
    is_excalidraw_mcp_provider_available,
)
from app.services.artifacts.persistence import (
    list_artifacts_by_message_ids,
    persist_artifact_responses,
)
from app.services.mcp.artifact import Artifact, sanitize_artifact

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
EXCALIDRAW_SERVER_ID = (settings.excalidraw_mcp_server_id or "excalidraw").strip() or "excalidraw"
EXCALIDRAW_EXPORT_TOOL = (settings.excalidraw_mcp_export_tool or "export_to_excalidraw").strip() or "export_to_excalidraw"
EXCALIDRAW_CREATE_VIEW_TOOL = (settings.excalidraw_mcp_create_view_tool or "create_view").strip() or "create_view"
EXCALIDRAW_INTENT_PATTERN = re.compile(
    r"\b(excalidraw|diagram|flowchart|whiteboard|wireframe|mind\s*map|draw|drawing|chart|schema)\b"
    r"|(?:\bs[ơo]\s*d[oồ]\b)"
    r"|(?:sơ\s*đồ)"
    r"|(?:\bv[eẽ]\b)"
    r"|(?:\bve\b)",
    re.IGNORECASE,
)
JSON_FENCE_PATTERN = re.compile(r"```(?:json|excalidraw)?\s*([\s\S]*?)```", re.IGNORECASE)


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


def _load_message_artifacts(payload: str | None) -> list[dict]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("Failed to decode message artifact payload.", exc_info=True)
        return []
    if not isinstance(data, dict):
        return []
    artifacts = data.get("artifacts") or []
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _load_message_tool_results(payload: str | None) -> list[dict]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("Failed to decode message tool-result payload.", exc_info=True)
        return []
    if not isinstance(data, dict):
        return []
    tool_results = data.get("tool_results") or []
    return [tool_result for tool_result in tool_results if isinstance(tool_result, dict)]


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
    assistant_message_ids = [int(m.id) for m in rows if m.role == "assistant" and getattr(m, "id", None) is not None]
    citations_by_request = _build_citations_by_request(db, assistant_request_ids)
    retrieval_by_request = _build_retrieval_by_request(db, assistant_request_ids)
    artifacts_by_message = list_artifacts_by_message_ids(db, assistant_message_ids)
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
            "artifacts": (
                artifacts_by_message.get(m.id)
                or _load_message_artifacts(m.__dict__.get("image_payload_json"))
                if m.role == "assistant"
                else []
            ),
            "tool_results": _load_message_tool_results(m.__dict__.get("image_payload_json")) if m.role == "assistant" else [],
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
    mcp_tool_prompt: str | None = None,
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

    if mcp_tool_prompt:
        sections.append(mcp_tool_prompt)

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


def _build_start_event_metadata(prepared: PreparedChatTurn) -> dict:
    retrieval_result = prepared.retrieval_result or {}
    safe_sources: list[dict] = []
    for source in prepared.sources or []:
        if not isinstance(source, dict):
            continue
        safe_sources.append(
            {
                "document_id": source.get("document_id"),
                "chunk_id": source.get("chunk_id"),
                "title": source.get("title"),
                "source_type": source.get("source_type"),
                "rank": source.get("rank"),
            }
        )

    return {
        "request_id": prepared.request_id,
        "rag_mode": retrieval_result.get("rag_mode"),
        "retrieval_scope": retrieval_result.get("retrieval_scope"),
        "sources": safe_sources,
        "has_web_search": bool((prepared.web_search_result or {}).get("used"))
        or bool(retrieval_result.get("web_search_used")),
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
    mcp_client_manager=None,
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
    mcp_tool_prompt = _build_mcp_tool_prompt(mcp_client_manager)
    system_prompt = _compose_system_prompt(
        summary_text,
        retrieval_result,
        knowledge_document_id=knowledge_document_id,
        web_search_result=web_search_result,
        knowledge_base_active=knowledge_base_active,
        mcp_tool_prompt=mcp_tool_prompt,
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


def _build_mcp_tool_prompt(mcp_client_manager) -> str | None:
    """Build a system prompt section describing available MCP tools.

    Instructs the LLM to always include a text description of tool results
    so text-only clients get a useful response. Returns None when MCP is
    disabled or no tools are registered.
    """
    if not mcp_client_manager:
        return None
    try:
        enabled = getattr(mcp_client_manager, "enabled", False)
        if not enabled:
            return None
        config = getattr(mcp_client_manager, "config", None)
        if not config:
            return None
        servers = getattr(config, "servers", []) or []
        active_servers = [s for s in servers if getattr(s, "enabled", False)]
        if not active_servers:
            return None

        lines = [
            "You have access to external tools that can create diagrams, fetch data, or perform actions.",
            "When using a tool, always include a text description of the result in your response.",
            "For Excalidraw diagrams, use create_view with a JSON array of Excalidraw elements; export_to_excalidraw is only for app-side exports.",
        ]
        for server in active_servers:
            sid = getattr(server, "id", "unknown")
            label = getattr(server, "label", sid)
            caps = getattr(server, "artifact_capabilities", [])
            lines.append(f"- {label}: capabilities: {', '.join(caps) if caps else 'generic tool'}")
        lines.append(
            "If a tool execution fails, explain what went wrong in natural language. "
            "Never leave your reply empty after a tool invocation."
        )
        return "\n\n".join(lines)
    except Exception:
        logger.debug("Failed to build MCP system prompt section", exc_info=True)
        return None


def _run_async_sync(coro_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro_factory()))
        return future.result()


def _is_excalidraw_request(user_message: str, mcp_client_manager) -> bool:
    if not mcp_client_manager or not is_diagram_intent(user_message):
        return False
    return is_excalidraw_mcp_provider_available(
        mcp_client_manager,
        server_id=EXCALIDRAW_SERVER_ID,
        create_view_tool=EXCALIDRAW_CREATE_VIEW_TOOL,
        export_tool=EXCALIDRAW_EXPORT_TOOL,
        provider_mode=settings.excalidraw_artifact_provider,
    )


def _should_use_native_excalidraw_artifact(user_message: str | None) -> bool:
    return bool(settings.enable_excalidraw_artifacts and is_diagram_intent(user_message))


def _configure_diagram_generation(prepared: PreparedChatTurn) -> None:
    prepared.request_kwargs["system"] = build_diagram_system_prompt()
    diagram_model = (settings.excalidraw_diagram_model or "").strip()
    if diagram_model:
        prepared.request_kwargs["model"] = diagram_model


def _json_candidates_from_text(text: str) -> list[object]:
    candidates: list[object] = []
    for match in JSON_FENCE_PATTERN.finditer(text or ""):
        fenced = match.group(1).strip()
        if not fenced:
            continue
        try:
            candidates.append(json.loads(fenced))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(parsed)
    return candidates


def _partial_json_array_objects(text: str) -> list[dict]:
    return partial_json_array_objects(text)


def _extract_partial_excalidraw_scene(text: str) -> tuple[dict, list[dict]] | None:
    raw_elements = _partial_json_array_objects(text)
    try:
        elements = normalize_excalidraw_elements(
            raw_elements,
            max_payload_bytes=settings.excalidraw_artifact_max_bytes,
        )
    except ExcalidrawValidationError:
        elements = raw_elements
    if not elements:
        return None
    scene = build_excalidraw_scene(elements)
    return scene, elements


def _scene_from_candidate(candidate: object) -> tuple[dict, list[dict]] | None:
    if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
        try:
            elements = normalize_excalidraw_elements(
                candidate,
                max_payload_bytes=settings.excalidraw_artifact_max_bytes,
            )
        except ExcalidrawValidationError:
            return None
        scene = build_excalidraw_scene(elements)
        return scene, elements

    if not isinstance(candidate, dict):
        return None
    raw_elements = candidate.get("elements")
    if not isinstance(raw_elements, list) or not all(isinstance(item, dict) for item in raw_elements):
        return None

    try:
        elements = normalize_excalidraw_elements(
            raw_elements,
            max_payload_bytes=settings.excalidraw_artifact_max_bytes,
        )
    except ExcalidrawValidationError:
        return None
    scene = dict(candidate)
    scene.setdefault("type", "excalidraw")
    scene.setdefault("version", 2)
    scene.setdefault("source", "DominicChatbot")
    scene.setdefault("appState", {"viewBackgroundColor": "#ffffff"})
    scene.setdefault("files", {})
    scene["elements"] = elements
    return scene, elements


def _extract_excalidraw_scene(text: str) -> tuple[dict, list[dict]] | None:
    for candidate in _json_candidates_from_text(text):
        scene = _scene_from_candidate(candidate)
        if scene is not None:
            return scene
    return None


def _excalidraw_stream_artifact_response(scene: dict, *, request_id: str, sequence: int) -> dict:
    return {
        "id": f"stream_excalidraw_{request_id}",
        "type": "excalidraw",
        "title": "Excalidraw create_view",
        "content": json.dumps(scene, ensure_ascii=False, separators=(",", ":")),
        "url": None,
        "preview_url": None,
        "metadata": {
            "tool_server": EXCALIDRAW_SERVER_ID,
            "tool_name": EXCALIDRAW_CREATE_VIEW_TOOL,
            "mcp_app_tool": "create_view",
            "render_mode": "inline_create_view_stream",
            "streaming": True,
            "sequence": sequence,
            "element_count": len(scene.get("elements") or []),
        },
    }


def _excalidraw_text_element(
    *,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    font_size: int = 20,
) -> dict:
    element_id = uuid4().hex[:20]
    return {
        "id": element_id,
        "type": "text",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": 100000 + len(text),
        "version": 1,
        "versionNonce": 200000 + len(text),
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
        "text": text,
        "fontSize": font_size,
        "fontFamily": 1,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": None,
        "originalText": text,
        "lineHeight": 1.25,
    }


def _excalidraw_rectangle_element(*, x: int, y: int, width: int, height: int) -> dict:
    return {
        "id": uuid4().hex[:20],
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "#e7f5ff",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": 300000 + width + height,
        "version": 1,
        "versionNonce": 400000 + width + height,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }


def _excalidraw_ellipse_element(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    background_color: str = "#ffffff",
) -> dict:
    element = _excalidraw_rectangle_element(x=x, y=y, width=width, height=height)
    element["type"] = "ellipse"
    element["backgroundColor"] = background_color
    element["roundness"] = None
    return element


def _excalidraw_line_element(
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    dashed: bool = False,
    arrow: bool = False,
) -> dict:
    dx = x2 - x1
    dy = y2 - y1
    return {
        "id": uuid4().hex[:20],
        "type": "arrow" if arrow else "line",
        "x": x1,
        "y": y1,
        "width": dx,
        "height": dy,
        "angle": 0,
        "strokeColor": "#495057",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": 500000 + abs(dx) + abs(dy),
        "version": 1,
        "versionNonce": 600000 + abs(dx) + abs(dy),
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow" if arrow else None,
    }


def _looks_like_sales_use_case_request(user_message: str, ai_content: str) -> bool:
    raw_text = f"{user_message or ''}\n{ai_content or ''}"
    text = unicodedata.normalize("NFKD", raw_text).encode("ascii", "ignore").decode("ascii").lower()
    has_use_case = "use case" in text or "ca su dung" in text
    has_sales = "ban hang" in text or "sales" in text or "san pham" in text
    return has_use_case and has_sales


def _build_sales_use_case_excalidraw_scene() -> tuple[dict, list[dict]]:
    elements: list[dict] = []

    def add_text(text: str, x: int, y: int, width: int, height: int, font_size: int = 16):
        elements.append(_excalidraw_text_element(text=text, x=x, y=y, width=width, height=height, font_size=font_size))

    def add_actor(label: str, x: int, y: int):
        elements.append(_excalidraw_rectangle_element(x=x, y=y, width=150, height=54))
        add_text(label, x + 10, y + 13, 130, 24, 15)

    def add_use_case(label: str, x: int, y: int):
        elements.append(_excalidraw_ellipse_element(x=x, y=y, width=220, height=58, background_color="#fff9db"))
        add_text(label, x + 20, y + 15, 180, 28, 15)

    def add_line(x1: int, y1: int, x2: int, y2: int, dashed: bool = False, arrow: bool = False):
        elements.append(_excalidraw_line_element(x1=x1, y1=y1, x2=x2, y2=y2, dashed=dashed, arrow=arrow))

    elements.append(_excalidraw_rectangle_element(x=230, y=40, width=810, height=650))
    elements[-1]["backgroundColor"] = "transparent"
    add_text("H\u1ec7 th\u1ed1ng b\u00e1n h\u00e0ng", 515, 58, 240, 30, 20)

    use_cases = {
        "login": ("\u0110\u0103ng nh\u1eadp", 520, 105),
        "view": ("Xem s\u1ea3n ph\u1ea9m", 310, 190),
        "search": ("T\u00ecm ki\u1ebfm s\u1ea3n ph\u1ea9m", 650, 190),
        "cart": ("Th\u00eam v\u00e0o gi\u1ecf h\u00e0ng", 310, 285),
        "checkout": ("Thanh to\u00e1n", 650, 285),
        "history": ("Xem l\u1ecbch s\u1eed \u0111\u01a1n h\u00e0ng", 310, 380),
        "product": ("Qu\u1ea3n l\u00fd s\u1ea3n ph\u1ea9m", 650, 380),
        "order": ("Qu\u1ea3n l\u00fd \u0111\u01a1n h\u00e0ng", 310, 475),
        "user": ("Qu\u1ea3n l\u00fd ng\u01b0\u1eddi d\u00f9ng", 650, 475),
        "success": ("Thanh to\u00e1n th\u00e0nh c\u00f4ng", 650, 570),
        "failed": ("Thanh to\u00e1n th\u1ea5t b\u1ea1i", 310, 570),
    }

    add_actor("Kh\u00e1ch h\u00e0ng", 35, 250)
    add_actor("Nh\u00e2n vi\u00ean", 35, 455)
    add_actor("Qu\u1ea3n tr\u1ecb vi\u00ean", 1110, 360)

    add_line(185, 277, 310, 219)
    add_line(185, 277, 650, 219)
    add_line(185, 277, 310, 314)
    add_line(185, 277, 650, 314)
    add_line(185, 277, 310, 409)
    add_line(185, 482, 310, 504)
    add_line(185, 482, 310, 219)
    add_line(185, 482, 650, 219)
    add_line(1110, 387, 870, 409)
    add_line(1110, 387, 530, 504)
    add_line(1110, 387, 870, 504)
    add_line(420, 314, 575, 134, dashed=True, arrow=True)
    add_line(760, 314, 630, 134, dashed=True, arrow=True)
    add_line(760, 314, 760, 570, dashed=True, arrow=True)
    add_line(760, 314, 420, 570, dashed=True, arrow=True)

    for label, x, y in use_cases.values():
        add_use_case(label, x, y)

    add_text("<<include>>", 455, 205, 95, 24, 12)
    add_text("<<include>>", 665, 195, 95, 24, 12)
    add_text("<<extend>>", 770, 430, 95, 24, 12)
    add_text("<<extend>>", 500, 430, 95, 24, 12)

    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "Dominic MCP",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    return scene, elements


def _build_fallback_excalidraw_scene(user_message: str, ai_content: str) -> tuple[dict, list[dict]]:
    if _looks_like_sales_use_case_request(user_message, ai_content):
        return _build_sales_use_case_excalidraw_scene()

    title = (user_message or "Excalidraw diagram").strip()
    summary = re.sub(r"\s+", " ", (ai_content or "").strip())
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."
    if not summary:
        summary = "Generated from the chat request."

    elements = [
        _excalidraw_rectangle_element(x=0, y=0, width=520, height=110),
        _excalidraw_text_element(text=title[:120], x=20, y=28, width=480, height=54, font_size=24),
        _excalidraw_rectangle_element(x=0, y=170, width=520, height=150),
        _excalidraw_text_element(text=summary, x=24, y=200, width=472, height=90, font_size=18),
    ]
    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "Dominic MCP",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    return scene, elements


def _tool_allowed(mcp_client_manager, tool_name: str) -> bool:
    checker = getattr(mcp_client_manager, "is_tool_allowed", None)
    if callable(checker):
        try:
            return bool(checker(EXCALIDRAW_SERVER_ID, tool_name))
        except Exception:
            logger.debug("Failed to check MCP tool allowlist", exc_info=True)
            return False
    return True


def _invoke_mcp_tool_sync(mcp_client_manager, tool_name: str, arguments: dict, prepared: PreparedChatTurn):
    async def call_tool():
        return await mcp_client_manager.invoke_tool(
            EXCALIDRAW_SERVER_ID,
            tool_name,
            arguments,
            user=prepared.username,
            session=str(prepared.session_id),
            turn_id=prepared.request_id,
        )

    return _run_async_sync(call_tool)


def _normalize_mcp_artifacts(mcp_client_manager, tool_result) -> list:
    if getattr(tool_result, "status", None) != "success":
        return []
    max_content_bytes = getattr(getattr(mcp_client_manager, "global_config", None), "max_artifact_content_bytes", None)
    artifacts = normalize_tool_result(
        EXCALIDRAW_SERVER_ID,
        getattr(tool_result, "tool_name", EXCALIDRAW_EXPORT_TOOL),
        getattr(tool_result, "raw_content", None),
        max_content_bytes=max_content_bytes,
    )
    try:
        tool_result.artifact_ids = [artifact.id for artifact in artifacts]
    except Exception:
        logger.debug("Failed to attach MCP artifact ids to tool result", exc_info=True)
    return artifacts


def _checkpoint_id_from_tool_result(tool_result) -> str | None:
    raw = getattr(tool_result, "raw_content", None)
    if not isinstance(raw, dict):
        raw_dump = getattr(raw, "model_dump", None)
        if callable(raw_dump):
            try:
                raw = raw_dump()
            except Exception:
                raw = None
    if not isinstance(raw, dict):
        return None
    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        checkpoint_id = structured.get("checkpointId")
        if isinstance(checkpoint_id, str) and checkpoint_id.strip():
            return checkpoint_id.strip()
    return None


def _annotate_create_view_artifacts(artifacts: list, tool_result, element_count: int) -> list:
    checkpoint_id = _checkpoint_id_from_tool_result(tool_result)
    for artifact in artifacts:
        try:
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            metadata.update({
                "tool_server": EXCALIDRAW_SERVER_ID,
                "tool_name": getattr(tool_result, "tool_name", EXCALIDRAW_CREATE_VIEW_TOOL),
                "mcp_app_tool": "create_view",
                "render_mode": "inline_create_view",
                "element_count": element_count,
            })
            if checkpoint_id:
                metadata["checkpoint_id"] = checkpoint_id
            artifact.metadata = metadata
            artifact.title = "Excalidraw create_view"
        except Exception:
            logger.debug("Failed to annotate Excalidraw create_view artifact", exc_info=True)
    return artifacts


def _fallback_scene_artifacts(mcp_client_manager, tool_name: str, scene: dict) -> list:
    max_content_bytes = getattr(getattr(mcp_client_manager, "global_config", None), "max_artifact_content_bytes", None)
    return normalize_tool_result(
        EXCALIDRAW_SERVER_ID,
        tool_name,
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(scene, ensure_ascii=False, separators=(",", ":")),
                }
            ]
        },
        max_content_bytes=max_content_bytes,
    )


def _native_excalidraw_artifact(
    *,
    artifact_id: str,
    title: str,
    request_id: str,
    elements: list[dict],
) -> list[Artifact]:
    artifact_data = artifact_response_from_elements(
        elements,
        artifact_id=artifact_id,
        title=title,
        request_id=request_id,
        streaming=False,
    )
    artifact = Artifact(
        id=artifact_data["id"],
        type=artifact_data["type"],
        title=artifact_data["title"],
        mime_type="application/json",
        content=artifact_data["content"],
        url=None,
        preview_url=None,
        metadata=artifact_data["metadata"],
        tool_server_id="native",
        tool_name="excalidraw_artifact",
    )
    sanitized = sanitize_artifact(
        artifact,
        max_content_bytes=settings.excalidraw_artifact_max_bytes,
    )
    return [sanitized] if sanitized is not None else []


def _maybe_create_excalidraw_artifacts(
    mcp_client_manager,
    prepared: PreparedChatTurn,
    ai_content: str,
) -> tuple[list, list]:
    if not _is_excalidraw_request(prepared.user_message, mcp_client_manager):
        return [], []

    scene_and_elements = _extract_excalidraw_scene(ai_content)
    if not scene_and_elements or not scene_and_elements[1]:
        scene_and_elements = _build_fallback_excalidraw_scene(prepared.user_message, ai_content)
    scene, elements = scene_and_elements
    tool_results: list = []

    if _tool_allowed(mcp_client_manager, EXCALIDRAW_CREATE_VIEW_TOOL):
        result = _invoke_mcp_tool_sync(
            mcp_client_manager,
            EXCALIDRAW_CREATE_VIEW_TOOL,
            {"elements": json.dumps(elements, ensure_ascii=False, separators=(",", ":"))},
            prepared,
        )
        tool_results.append(result)
        artifacts = _normalize_mcp_artifacts(mcp_client_manager, result)
        if artifacts:
            _annotate_create_view_artifacts(artifacts, result, len(elements))
            return artifacts, tool_results
        artifacts = _fallback_scene_artifacts(mcp_client_manager, EXCALIDRAW_CREATE_VIEW_TOOL, scene)
        _annotate_create_view_artifacts(artifacts, result, len(elements))
        try:
            result.artifact_ids = [artifact.id for artifact in artifacts]
        except Exception:
            logger.debug("Failed to attach fallback MCP artifact ids", exc_info=True)
        if artifacts:
            return artifacts, tool_results

    if _tool_allowed(mcp_client_manager, EXCALIDRAW_EXPORT_TOOL):
        result = _invoke_mcp_tool_sync(
            mcp_client_manager,
            EXCALIDRAW_EXPORT_TOOL,
            {"json": json.dumps(scene, ensure_ascii=False, separators=(",", ":"))},
            prepared,
        )
        tool_results.append(result)
        artifacts = _normalize_mcp_artifacts(mcp_client_manager, result)
        if artifacts:
            return artifacts, tool_results
        if getattr(result, "status", None) == "success":
            artifacts = _fallback_scene_artifacts(mcp_client_manager, EXCALIDRAW_EXPORT_TOOL, scene)
            try:
                result.artifact_ids = [artifact.id for artifact in artifacts]
            except Exception:
                logger.debug("Failed to attach fallback MCP artifact ids", exc_info=True)
            if artifacts:
                return artifacts, tool_results

    if tool_results:
        fallback_tool_name = getattr(tool_results[-1], "tool_name", EXCALIDRAW_EXPORT_TOOL)
        artifacts = _fallback_scene_artifacts(mcp_client_manager, fallback_tool_name, scene)
        try:
            tool_results[-1].artifact_ids = [artifact.id for artifact in artifacts]
        except Exception:
            logger.debug("Failed to attach final fallback MCP artifact ids", exc_info=True)
        if artifacts:
            return artifacts, tool_results

    return [], tool_results


def _generate_excalidraw_artifacts(
    mcp_client_manager,
    prepared: PreparedChatTurn,
    ai_content: str,
    *,
    artifact_id: str,
    title: str,
    elements: list[dict] | None = None,
) -> ArtifactGenerationResult:
    request = ArtifactGenerationRequest(
        kind="excalidraw",
        user_message=prepared.user_message,
        request_id=prepared.request_id,
        artifact_id=artifact_id,
        title=title,
        model_output=ai_content,
        elements=elements,
        username=prepared.username,
        session_id=prepared.session_id,
        context=prepared,
    )

    def native_generate(provider_request: ArtifactGenerationRequest) -> ArtifactGenerationResult:
        if not provider_request.elements:
            return ArtifactGenerationResult(provider_id="native_llm_excalidraw")
        return ArtifactGenerationResult(
            provider_id="native_llm_excalidraw",
            artifacts=_native_excalidraw_artifact(
                artifact_id=provider_request.artifact_id,
                title=provider_request.title,
                request_id=provider_request.request_id,
                elements=provider_request.elements,
            ),
        )

    def mcp_generate(provider_request: ArtifactGenerationRequest) -> ArtifactGenerationResult:
        artifacts, tool_results = _maybe_create_excalidraw_artifacts(
            mcp_client_manager,
            provider_request.context,
            provider_request.model_output,
        )
        return ArtifactGenerationResult(
            provider_id="mcp_excalidraw",
            artifacts=artifacts,
            tool_results=tool_results,
            fallback_used=bool(artifacts and tool_results and getattr(tool_results[-1], "status", None) != "success"),
        )

    native_provider = CallableArtifactProvider(
        provider_id="native_llm_excalidraw",
        kind="excalidraw",
        generate=native_generate,
        is_available=lambda provider_request: bool(provider_request.elements),
    )
    mcp_provider = CallableArtifactProvider(
        provider_id="mcp_excalidraw",
        kind="excalidraw",
        generate=mcp_generate,
        is_available=lambda _provider_request: _is_excalidraw_request(prepared.user_message, mcp_client_manager),
    )
    service = ArtifactGenerationService(
        [mcp_provider],
        fallback_provider=native_provider,
    )
    return service.generate(request)


def _reply_for_excalidraw_artifact(ai_content: str, artifacts: list) -> str:
    if not artifacts:
        return ai_content
    return "I created the Excalidraw diagram. Open the artifact below to view or edit it."


def _should_suppress_excalidraw_stream_text(prepared: PreparedChatTurn, text: str, mcp_client_manager) -> bool:
    if not _is_excalidraw_request(prepared.user_message, mcp_client_manager):
        return False
    stripped = (text or "").lstrip()
    if stripped.startswith("[") or stripped.startswith("{") or stripped.startswith("```"):
        return True
    return bool(_extract_partial_excalidraw_scene(text))


def _artifacts_to_response(artifacts: list | None) -> list[dict] | None:
    """Convert internal Artifact models to public-safe response dicts.

    Returns None when empty so the field is omitted from JSON.
    Only includes artifacts that passed sanitization (safe=True).
    """
    if not artifacts:
        return None
    result: list[dict] = []
    for art in artifacts:
        if not hasattr(art, "safe") or not getattr(art, "safe", False):
            continue
        item = {
            "id": getattr(art, "id", ""),
            "type": getattr(art, "type", "generic_tool_result"),
            "title": getattr(art, "title", ""),
            "url": getattr(art, "url", None),
            "preview_url": getattr(art, "preview_url", None),
            "metadata": getattr(art, "metadata", {}),
        }
        content = getattr(art, "content", None)
        if content is not None:
            item["content"] = content
        result.append(item)
    return result or None


def _tool_results_to_response(tool_results: list | None) -> list[dict] | None:
    """Convert internal McpToolResult models to public-safe response dicts.

    Returns None when empty so the field is omitted from JSON.
    """
    if not tool_results:
        return None
    result: list[dict] = []
    for tr in tool_results:
        result.append({
            "tool_server_id": getattr(tr, "server_id", ""),
            "tool_name": getattr(tr, "tool_name", ""),
            "status": getattr(tr, "status", "error"),
            "duration_ms": getattr(tr, "duration_ms", 0),
            "artifact_ids": getattr(tr, "artifact_ids", []),
        })
    return result or None


def _finalize_chat_turn(
    db: Session,
    prepared: PreparedChatTurn,
    *,
    ai_content: str,
    input_tokens: int,
    output_tokens: int,
    mcp_artifacts: list | None = None,
    mcp_tool_results: list | None = None,
    skip_answer_guardrails: bool = False,
) -> dict:
    sources = list(prepared.sources)
    retrieval_result = prepared.retrieval_result
    web_search_result = prepared.web_search_result

    if skip_answer_guardrails:
        answer_policy = "artifact"
    else:
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

    artifacts_response = _artifacts_to_response(mcp_artifacts)
    tool_results_response = _tool_results_to_response(mcp_tool_results)

    assistant_message = crud_chat.create_message(
        db=db,
        role="assistant",
        sender_username=prepared.username,
        session_id=prepared.session_id,
        content=ai_content,
        request_id=prepared.request_id,
        sources=persisted_web_sources or None,
        assistant_meta=assistant_meta,
        artifacts=artifacts_response,
        tool_results=tool_results_response,
        input_tokens=0,
        output_tokens=output_tokens,
        status="success",
    )
    if artifacts_response is not None:
        try:
            persist_artifact_responses(
                db,
                session_id=prepared.session_id,
                message_id=assistant_message.id,
                artifacts=artifacts_response,
            )
        except Exception:
            logger.debug("First-class artifact persistence failed; message payload remains authoritative", exc_info=True)

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

    result = {
        "reply": ai_content,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "request_id": prepared.request_id,
        "sources": sources,
        "assistant_meta": assistant_meta,
        "retrieval": _build_retrieval_payload(retrieval_result)
        or _build_web_search_payload(web_search_result, answer_policy),
    }
    if artifacts_response is not None:
        result["artifacts"] = artifacts_response
    if tool_results_response is not None:
        result["tool_results"] = tool_results_response
    return result


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
    mcp_client_manager=None,
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
            mcp_client_manager=mcp_client_manager,
        )

        logger.info(
            "LiteLLM call username=%s session_id=%s model=%s messages=%d images=%d",
            username, session_id, llm_provider.resolve_model(model), len(prepared.request_kwargs["messages"]), len(images or []),
        )
        native_diagram = _should_use_native_excalidraw_artifact(prepared.user_message)
        if native_diagram:
            _configure_diagram_generation(prepared)
        llm_result = llm_provider.complete(**prepared.request_kwargs)
        if native_diagram:
            try:
                elements = parse_final_elements(
                    llm_result["text"],
                    max_payload_bytes=settings.excalidraw_artifact_max_bytes,
                )
            except ExcalidrawValidationError:
                elements = repair_final_elements(
                    llm_result["text"],
                    max_payload_bytes=settings.excalidraw_artifact_max_bytes,
                )
            artifact_result = _generate_excalidraw_artifacts(
                mcp_client_manager,
                prepared,
                llm_result["text"],
                artifact_id=artifact_id_for_request(prepared.request_id),
                title="Excalidraw diagram",
                elements=elements,
            )
            mcp_artifacts = artifact_result.artifacts
            mcp_tool_results = artifact_result.tool_results
        else:
            artifact_result = _generate_excalidraw_artifacts(
                mcp_client_manager,
                prepared,
                llm_result["text"],
                artifact_id=artifact_id_for_request(prepared.request_id),
                title="Excalidraw diagram",
            )
            mcp_artifacts = artifact_result.artifacts
            mcp_tool_results = artifact_result.tool_results
        return _finalize_chat_turn(
            db,
            prepared,
            ai_content=_reply_for_excalidraw_artifact(llm_result["text"], mcp_artifacts),
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            mcp_artifacts=mcp_artifacts,
            mcp_tool_results=mcp_tool_results,
            skip_answer_guardrails=native_diagram and bool(mcp_artifacts),
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
    mcp_client_manager=None,
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
            mcp_client_manager=mcp_client_manager,
        )

        yield {"event": "start", "data": _build_start_event_metadata(prepared)}

        native_diagram = _should_use_native_excalidraw_artifact(prepared.user_message)
        artifact_id = artifact_id_for_request(prepared.request_id)
        artifact_title = "Excalidraw diagram"
        if native_diagram:
            _configure_diagram_generation(prepared)
            yield {
                "event": "artifact_start",
                "data": artifact_start_event(
                    artifact_id=artifact_id,
                    title=artifact_title,
                    request_id=prepared.request_id,
                ),
            }

        stream_result: dict | None = None
        streamed_text = ""
        last_streamed_element_count = 0
        artifact_sequence = 0
        suppress_text_stream = native_diagram or _is_excalidraw_request(prepared.user_message, mcp_client_manager)
        for chunk in llm_provider.stream_complete(**prepared.request_kwargs):
            chunk_type = chunk.get("type")
            if chunk_type == "delta":
                delta_text = chunk.get("text") or ""
                if delta_text:
                    streamed_text += delta_text
                    partial_scene = _extract_partial_excalidraw_scene(streamed_text) if suppress_text_stream else None
                    if partial_scene is not None:
                        scene, elements = partial_scene
                        if len(elements) > last_streamed_element_count:
                            last_streamed_element_count = len(elements)
                            artifact_sequence += 1
                            if native_diagram:
                                yield {
                                    "event": "artifact_delta",
                                    "data": artifact_delta_event(
                                        artifact_id=artifact_id,
                                        title=artifact_title,
                                        request_id=prepared.request_id,
                                        elements_partial=streamed_text,
                                        elements=elements,
                                        sequence=artifact_sequence,
                                    ),
                                }
                            else:
                                yield {
                                    "event": "artifact_delta",
                                    "data": {
                                        "request_id": prepared.request_id,
                                        "artifact": _excalidraw_stream_artifact_response(
                                            scene,
                                            request_id=prepared.request_id,
                                            sequence=artifact_sequence,
                                        ),
                                    },
                                }
                    if not suppress_text_stream and not _should_suppress_excalidraw_stream_text(prepared, streamed_text, mcp_client_manager):
                        yield {
                            "event": "delta",
                            "data": {"text": delta_text, "request_id": prepared.request_id},
                        }
            elif chunk_type == "complete":
                stream_result = chunk

        if stream_result is None:
            raise RuntimeError("Streaming provider completed without a final payload.")

        final_elements: list[dict] | None = None
        if native_diagram:
            try:
                final_elements = parse_final_elements(
                    stream_result["text"],
                    max_payload_bytes=settings.excalidraw_artifact_max_bytes,
                )
            except ExcalidrawValidationError:
                try:
                    final_elements = repair_final_elements(
                        stream_result["text"],
                        max_payload_bytes=settings.excalidraw_artifact_max_bytes,
                    )
                except ExcalidrawValidationError as repair_error:
                    yield {
                        "event": "artifact_error",
                        "data": artifact_error_event(
                            artifact_id=artifact_id,
                            request_id=prepared.request_id,
                            message=str(repair_error),
                        ),
                    }

            if final_elements:
                if len(final_elements) > last_streamed_element_count:
                    artifact_sequence += 1
                    yield {
                        "event": "artifact_delta",
                        "data": artifact_delta_event(
                            artifact_id=artifact_id,
                            title=artifact_title,
                            request_id=prepared.request_id,
                            elements_partial=stream_result["text"],
                            elements=final_elements,
                            sequence=artifact_sequence,
                        ),
                    }
                yield {
                    "event": "artifact_done",
                    "data": artifact_done_event(
                        artifact_id=artifact_id,
                        title=artifact_title,
                        request_id=prepared.request_id,
                        elements=final_elements,
                    ),
                }
        else:
            final_scene_and_elements = _extract_excalidraw_scene(stream_result["text"])
            if final_scene_and_elements and len(final_scene_and_elements[1]) > last_streamed_element_count:
                artifact_sequence += 1
                yield {
                    "event": "artifact_delta",
                    "data": {
                        "request_id": prepared.request_id,
                        "artifact": _excalidraw_stream_artifact_response(
                            final_scene_and_elements[0],
                            request_id=prepared.request_id,
                            sequence=artifact_sequence,
                        ),
                    },
                }

        artifact_result = _generate_excalidraw_artifacts(
            mcp_client_manager,
            prepared,
            stream_result["text"],
            artifact_id=artifact_id,
            title=artifact_title,
            elements=final_elements if native_diagram and final_elements else None,
        )
        mcp_artifacts = artifact_result.artifacts
        mcp_tool_results = artifact_result.tool_results
        final_result = _finalize_chat_turn(
            db,
            prepared,
            ai_content=(
                _reply_for_excalidraw_artifact(stream_result["text"], mcp_artifacts)
                if not native_diagram or mcp_artifacts
                else "I could not create a valid Excalidraw diagram from the model output."
            ),
            input_tokens=stream_result["input_tokens"],
            output_tokens=stream_result["output_tokens"],
            mcp_artifacts=mcp_artifacts,
            mcp_tool_results=mcp_tool_results,
            skip_answer_guardrails=native_diagram and bool(mcp_artifacts),
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
