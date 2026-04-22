import logging
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.crud import crud_chat
from app.crud import crud_knowledge
from app.services.retrieval_service import search_knowledge
from app.services import llm_provider
from app.services.llm_provider import LLMError

# Local aliases for readability (resolved from settings at module load)
CONTEXT_WINDOW_SIZE = settings.context_window_size
MAX_OUTPUT_TOKENS = settings.max_output_tokens
ROLLING_WINDOW_HOURS = settings.rolling_window_hours
SUMMARY_MAX_TOKENS = settings.summary_max_tokens
SUMMARY_TRIGGER_MESSAGES = settings.summary_trigger_messages
TOKEN_ESTIMATE_CHARS_PER_TOKEN = settings.token_estimate_chars_per_token

logger = logging.getLogger("uvicorn.error")


class ProviderRequestError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail

    @classmethod
    def from_llm_error(cls, e: LLMError) -> "ProviderRequestError":
        return cls(e.status_code, e.detail)


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
    assistant_request_ids = [m.request_id for m in rows if m.role == "assistant" and m.request_id]
    citations_by_request = _build_citations_by_request(db, assistant_request_ids)
    retrieval_by_request = _build_retrieval_by_request(db, assistant_request_ids)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "input_tokens": int(m.input_tokens or 0),
            "output_tokens": int(m.output_tokens or 0),
            "created_at": m.created_at,
            "request_id": m.request_id,
            "sources": citations_by_request.get(m.request_id, []) if m.role == "assistant" else [],
            "retrieval": retrieval_by_request.get(m.request_id) if m.role == "assistant" else None,
        }
        for m in rows
    ]


def _build_citations_by_request(db: Session, request_ids: list[str]) -> dict[str, list[dict]]:
    rows = crud_knowledge.list_answer_citations_by_request_ids(db, request_ids)
    grouped: dict[str, list[dict]] = {}
    for citation, document, _chunk in rows:
        grouped.setdefault(citation.request_id, []).append(
            {
                "document_id": citation.document_id,
                "chunk_id": citation.chunk_id,
                "title": document.title,
                "score": float(citation.score) if citation.score is not None else None,
                "snippet": citation.quoted_text or "",
                "source_uri": document.source_uri,
                "rank": citation.rank,
            }
        )
    return grouped


def _build_retrieval_by_request(db: Session, request_ids: list[str]) -> dict[str, dict]:
    rows = crud_knowledge.list_retrieval_events_by_request_ids(db, request_ids)
    grouped: dict[str, dict] = {}
    for event in rows:
        metadata = event.metadata_json or {}
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
        }
    return grouped


def _estimate_input_tokens(messages: list[dict], system_prompt: str | None = None) -> int:
    # Heuristic: ~4 chars/token for mixed Vietnamese/English text.
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
            "score": row.get("score"),
            "rerank_score": row.get("rerank_score"),
            "snippet": row.get("snippet") or "",
            "source_uri": row.get("source_uri"),
            "rank": index,
        }
        for index, row in enumerate(results, start=1)
    ]


def _build_knowledge_context(results: list[dict]) -> str:
    blocks: list[str] = []
    for index, row in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Source {index}] title={row['title']} document_id={row['document_id']} chunk_id={row['chunk_id']} score={float(row.get('score') or 0):.3f}",
                    row.get("content") or row.get("snippet") or "",
                ]
            )
        )
    return "\n\n".join(blocks)


def _pack_retrieval_results(results: list[dict]) -> tuple[list[dict], int]:
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
) -> str:
    retrieved_results = (retrieval_result or {}).get("packed_results") or (retrieval_result or {}).get("results") or []
    evidence_strength = (retrieval_result or {}).get("evidence_strength") or "none"
    answer_policy = (retrieval_result or {}).get("answer_policy") or "grounded"
    fallback_used = bool((retrieval_result or {}).get("fallback_used"))
    sections = [
        "You are Dominic, a helpful assistant.",
        "If knowledge-base sources are provided below, prioritize them for answers about uploaded documents.",
        "When using those sources, cite them inline as [Source 1], [Source 2], etc.",
        "Never fabricate sources or claim a document says something unless it is supported by the evidence block below.",
    ]

    if summary_text:
        sections.append(
            "Conversation memory summary (may omit small details). Use this as background context:\n"
            + summary_text
        )

    if retrieved_results:
        sections.append("Knowledge-base evidence for this turn:\n" + _build_knowledge_context(retrieved_results))

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


def _apply_answer_guardrails(
    ai_content: str,
    retrieval_result: dict | None,
    sources: list[dict],
    *,
    knowledge_document_id: int | None = None,
) -> tuple[str, list[dict], str]:
    answer_policy = _determine_answer_policy(
        retrieval_result,
        knowledge_document_id=knowledge_document_id,
    )

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


def handle_chat(
    db: Session,
    username: str,
    session_id: int,
    user_message: str,
    knowledge_document_id: int | None = None,
    images: list[str] | None = None,
    image_media_types: list[str] | None = None,
):
    user = crud_chat.get_user_by_username(db, username)
    if not user:
        raise ValueError(f"User '{username}' not found.")
    session = crud_chat.get_chat_session(db, username, session_id)
    if not session:
        raise ValueError("Session not found.")
    if knowledge_document_id is not None:
        knowledge_document = crud_knowledge.get_document(db, knowledge_document_id)
        if not knowledge_document or knowledge_document.owner_username != username:
            raise ValueError("Knowledge document not found.")

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
        summary_text, formatted_messages = _build_hybrid_context(db, username, session_id)
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
        retrieval_result["answer_policy"] = _determine_answer_policy(
            retrieval_result,
            knowledge_document_id=knowledge_document_id,
        )
        if retrieval_result.get("retrieval_id"):
            crud_knowledge.update_retrieval_event_metadata(
                db,
                retrieval_result["retrieval_id"],
                {
                    "packed_count": len(packed_results),
                    "packed_token_estimate": packed_token_estimate,
                    "answer_policy": retrieval_result["answer_policy"],
                },
            )

        sources = _build_sources(packed_results)
        system_prompt = _compose_system_prompt(
            summary_text,
            retrieval_result,
            knowledge_document_id=knowledge_document_id,
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
            "messages": request_messages,
            "system": system_prompt or None,
            "max_tokens": max_output_tokens,
        }
        # Attach images if provided (vision chat)
        if images and settings.llm_vision_enabled:
            request_kwargs["images"] = images
            request_kwargs["image_media_types"] = image_media_types or []

        logger.info(
            "LiteLLM call username=%s session_id=%s model=%s messages=%d images=%d",
            username, session_id, _get_model(), len(request_messages), len(images or []),
        )
        llm_result = llm_provider.complete(**request_kwargs)

        ai_content = llm_result["text"]
        in_tokens = llm_result["input_tokens"]
        out_tokens = llm_result["output_tokens"]

        ai_content, sources, answer_policy = _apply_answer_guardrails(
            ai_content,
            retrieval_result,
            sources,
            knowledge_document_id=knowledge_document_id,
        )
        retrieval_result["answer_policy"] = answer_policy
        if retrieval_result.get("retrieval_id"):
            crud_knowledge.update_retrieval_event_metadata(
                db,
                retrieval_result["retrieval_id"],
                {"answer_policy": answer_policy},
            )

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

        crud_knowledge.replace_answer_citations(
            db,
            request_id,
            citations=[
                {
                    "document_id": source["document_id"],
                    "chunk_id": source["chunk_id"],
                    "rank": source["rank"],
                    "score": source.get("score"),
                    "quoted_text": source.get("snippet") or "",
                }
                for source in sources
            ],
        )

        crud_chat.touch_chat_session(db, session_id)

        crud_chat.increment_user_tokens(db, username, in_tokens, out_tokens)

        return {
            "reply": ai_content,
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
            "request_id": request_id,
            "sources": sources,
            "retrieval": _build_retrieval_payload(retrieval_result),
        }

    except LLMError as e:
        logger.warning(
            "LLM call failed username=%s session_id=%s model=%s: %s",
            username, session_id, _get_model(), e.detail,
            exc_info=True,
        )
        crud_chat.update_message_tokens_and_status(
            db=db,
            message_id=user_msg.id,
            status="error",
            error_message=e.detail,
        )
        raise ProviderRequestError.from_llm_error(e) from e

    except Exception as e:
        crud_chat.update_message_tokens_and_status(
            db=db,
            message_id=user_msg.id,
            status="error",
            error_message=str(e),
        )
        raise
