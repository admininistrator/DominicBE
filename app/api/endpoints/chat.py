from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.core.database import get_database_error_message, is_unknown_database_error
from app.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    LoginRequest,
    LoginResponse,
    SessionCreateRequest,
    SessionMessageResponse,
    SessionResponse,
    UsageResponse,
)
from app.services.chat_service import (
    create_session,
    get_session_history,
    get_sessions,
    get_usage,
    handle_chat,
    login_user,
)

router = APIRouter()


def _raise_api_error(exc: Exception):
    if isinstance(exc, OperationalError):
        status_code = 503 if is_unknown_database_error(exc) else 500
        raise HTTPException(status_code=status_code, detail=get_database_error_message(exc))
    raise HTTPException(status_code=500, detail=str(exc))


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        result = login_user(db, request.username, request.password)
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        _raise_api_error(e)


@router.get("/usage/{username}", response_model=UsageResponse)
def get_user_usage(username: str, db: Session = Depends(get_db)):
    try:
        result = get_usage(db, username)
        return UsageResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _raise_api_error(e)


@router.post("/sessions", response_model=SessionResponse)
def create_chat_session(request: SessionCreateRequest, db: Session = Depends(get_db)):
    try:
        result = create_session(db, request.username, request.title)
        return SessionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _raise_api_error(e)


@router.get("/sessions/{username}", response_model=list[SessionResponse])
def list_user_sessions(username: str, db: Session = Depends(get_db)):
    try:
        result = get_sessions(db, username)
        return [SessionResponse(**row) for row in result]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _raise_api_error(e)


@router.get("/sessions/{username}/{session_id}/messages", response_model=list[SessionMessageResponse])
def get_messages_by_session(username: str, session_id: int, db: Session = Depends(get_db)):
    try:
        result = get_session_history(db, username, session_id)
        return [SessionMessageResponse(**row) for row in result]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        _raise_api_error(e)

@router.post("/", response_model=ChatResponse)
def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        result = handle_chat(db, request.username, request.session_id, request.message)
        return ChatResponse(
            success=True,
            reply=result["reply"],
            usage=result["usage"],
            request_id=result.get("request_id")
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _raise_api_error(e)
