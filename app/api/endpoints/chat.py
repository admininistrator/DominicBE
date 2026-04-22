from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, get_db
from app.models.chat_models import User
from app.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    SessionCreateRequest,
    SessionMessageResponse,
    SessionResponse,
    UsageResponse,
)
from app.services.chat_service import (
    ProviderRequestError,
    create_session,
    get_session_history,
    get_sessions,
    get_usage,
    handle_chat,
)

router = APIRouter()


def _assert_same_user(request_username: str | None, current_user: User) -> str:
    if request_username and request_username != current_user.username:
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{username}", response_model=list[SessionResponse])
def list_user_sessions(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        target_username = _assert_same_user(username, current_user)
        result = get_sessions(db, target_username)
        return [SessionResponse(**row) for row in result]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages", response_model=list[SessionMessageResponse])
def get_my_messages_by_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = get_session_history(db, current_user.username, session_id)
        return [SessionMessageResponse(**row) for row in result]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{username}/{session_id}/messages", response_model=list[SessionMessageResponse])
def get_messages_by_session(
    username: str,
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        target_username = _assert_same_user(username, current_user)
        result = get_session_history(db, target_username, session_id)
        return [SessionMessageResponse(**row) for row in result]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except ProviderRequestError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ChatResponse)
def send_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        username = _assert_same_user(request.username, current_user)
        result = handle_chat(
            db,
            username,
            request.session_id,
            request.message,
            knowledge_document_id=request.knowledge_document_id,
            images=request.images or None,
            image_media_types=request.image_media_types or None,
        )
        return ChatResponse(
            success=True,
            reply=result["reply"],
            usage=result["usage"],
            request_id=result.get("request_id"),
            sources=result.get("sources") or [],
            retrieval=result.get("retrieval"),
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
        raise HTTPException(status_code=500, detail=str(e))
