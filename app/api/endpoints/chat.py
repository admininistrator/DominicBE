import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, get_db
from app.api.error_handling import build_internal_server_error_payload, raise_internal_server_error
from app.core.logging import get_logger
from app.core.security import validate_username_policy
from app.models.chat_models import User
from app.services import llm_provider
from app.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    SessionCreateRequest,
    SessionRenameRequest,
    SessionMessageResponse,
    SessionResponse,
    UsageResponse,
)
from app.services.chat_service import (
    ProviderRequestError,
    create_session,
    delete_session,
    get_usage,
    get_session_history,
    get_sessions,
    handle_chat,
    handle_chat_stream,
    rename_session,
)

router = APIRouter()
logger = get_logger(__name__)
SESSION_HISTORY_MAX_LIMIT = 200


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _apply_session_history_pagination_headers(response: Response, pagination: dict):
    response.headers["X-Message-Pagination-Returned"] = str(pagination.get("returned", 0))
    response.headers["X-Message-Pagination-Has-More"] = "true" if pagination.get("has_more") else "false"

    limit = pagination.get("limit")
    if limit is not None:
        response.headers["X-Message-Pagination-Limit"] = str(limit)

    before_id = pagination.get("before_id")
    if before_id is not None:
        response.headers["X-Message-Pagination-Before-Id"] = str(before_id)

    next_before_id = pagination.get("next_before_id")
    if next_before_id is not None:
        response.headers["X-Message-Pagination-Next-Before-Id"] = str(next_before_id)

    response.headers["X-Message-Pagination-Skip"] = str(pagination.get("skip", 0))


def _assert_same_user(request_username: str | None, current_user: User) -> str:
    if request_username:
        try:
            normalized_request_username = validate_username_policy(
                request_username,
                field_name="Username",
                min_length=1,
                max_length=255,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if normalized_request_username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access your own account data.",
            )
    return current_user.username


@router.get("/usage/me", response_model=UsageResponse)
def get_my_usage(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        result = get_usage(db, current_user.username)
        return UsageResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise_internal_server_error(logger, action="chat.get_my_usage", exc=e)


@router.get("/usage/{username}", response_model=UsageResponse)
def get_user_usage(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        target_username = _assert_same_user(username, current_user)
        result = get_usage(db, target_username)
        return UsageResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise_internal_server_error(logger, action="chat.get_user_usage", exc=e)


@router.get("/models")
def get_chat_models(current_user: User = Depends(get_current_user)):
    del current_user
    try:
        return llm_provider.get_public_model_catalog()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/sessions", response_model=SessionResponse)
def create_chat_session(
    request: SessionCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        username = _assert_same_user(request.username, current_user)
        result = create_session(db, username, request.title)
        return SessionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise_internal_server_error(logger, action="chat.create_chat_session", exc=e)


@router.get("/sessions", response_model=list[SessionResponse])
def list_my_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        result = get_sessions(db, current_user.username)
        return [SessionResponse(**row) for row in result]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise_internal_server_error(logger, action="chat.list_my_sessions", exc=e)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_chat_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        deleted = delete_session(db, current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
    except HTTPException:
        raise
    except Exception as e:
        raise_internal_server_error(logger, action="chat.delete_chat_session", exc=e)


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
def rename_chat_session(
    session_id: int,
    request: SessionRenameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = rename_session(db, current_user.username, session_id, request.title)
        return SessionResponse(**result)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="chat.rename_chat_session", exc=e)


@router.get("/sessions/{session_id}/messages", response_model=list[SessionMessageResponse])
def get_my_messages_by_session(
    session_id: int,
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1, le=SESSION_HISTORY_MAX_LIMIT),
    before_id: int | None = Query(default=None, ge=1),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = get_session_history(
            db,
            current_user.username,
            session_id,
            skip=skip,
            limit=limit,
            before_id=before_id,
        )
        _apply_session_history_pagination_headers(response, result["pagination"])
        return [SessionMessageResponse(**row) for row in result["items"]]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise_internal_server_error(logger, action="chat.get_my_messages_by_session", exc=e)


@router.post("/", response_model=ChatResponse, response_model_exclude_none=True)
def send_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    fastapi_request: Request = None,
):
    try:
        username = _assert_same_user(request.username, current_user)
        mcp_client_manager = getattr(fastapi_request.app.state, "mcp_client_manager", None) if fastapi_request else None
        result = handle_chat(
            db,
            username,
            request.session_id,
            request.message,
            knowledge_document_id=request.knowledge_document_id,
            use_web_search=request.use_web_search,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            images=request.images or None,
            image_media_types=request.image_media_types or None,
            mcp_client_manager=mcp_client_manager,
        )
        return ChatResponse(
            success=True,
            reply=result["reply"],
            usage=result["usage"],
            request_id=result.get("request_id"),
            sources=result.get("sources") or [],
            assistant_meta=result.get("assistant_meta"),
            retrieval=result.get("retrieval"),
            artifacts=result.get("artifacts"),
            tool_results=result.get("tool_results"),
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise_internal_server_error(logger, action="chat.send_message", exc=e)


@router.post("/stream")
def stream_message(
    request_body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    username = _assert_same_user(request_body.username, current_user)
    mcp_client_manager = getattr(request.app.state, "mcp_client_manager", None) if request else None

    def event_stream():
        try:
            for event in handle_chat_stream(
                db,
                username,
                request_body.session_id,
                request_body.message,
                knowledge_document_id=request_body.knowledge_document_id,
                use_web_search=request_body.use_web_search,
                model=request_body.model,
                reasoning_effort=request_body.reasoning_effort,
                images=request_body.images or None,
                image_media_types=request_body.image_media_types or None,
                mcp_client_manager=mcp_client_manager,
            ):
                yield _sse_event(event["event"], event["data"])
        except PermissionError as e:
            yield _sse_event("error", {"status_code": 403, "detail": str(e)})
        except ProviderRequestError as e:
            yield _sse_event("error", {"status_code": e.status_code, "detail": e.detail})
        except ValueError as e:
            yield _sse_event("error", {"status_code": 400, "detail": str(e)})
        except HTTPException as e:
            yield _sse_event("error", {"status_code": e.status_code, "detail": e.detail})
        except Exception as e:
            yield _sse_event(
                "error",
                build_internal_server_error_payload(logger, action="chat.stream_message", exc=e),
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
