from sqlalchemy import func
from app.core.database import SessionLocal
from app.models.knowledge_models import RetrievalEvent, KnowledgeDocument, KnowledgeChunk, AuditLog
from app.models.chat_models import ChatSession, Message
from app.services.chat_service import get_session_history

def kind_of(value):
    if isinstance(value, dict): return "dict"
    if isinstance(value, list): return "list"
    if isinstance(value, str): return "str"
    if value is None: return "None"
    return type(value).__name__

def sample_of(value):
    text = repr(value)
    return text[:120] + ("..." if len(text) > 120 else "")

db = SessionLocal()
try:
    checks = [('retrieval_events.metadata_json', RetrievalEvent, 'metadata_json'), ('knowledge_documents.metadata_json', KnowledgeDocument, 'metadata_json'), ('knowledge_chunks.metadata_json', KnowledgeChunk, 'metadata_json'), ('audit_logs.detail_json', AuditLog, 'detail_json')]
    for label, model, attr in checks:
        column = getattr(model, attr)
        total = db.query(func.count(model.id)).scalar()
        nonnull = db.query(func.count(model.id)).filter(column.is_not(None)).scalar()
        rows = db.query(model).filter(column.is_not(None)).order_by(model.id).limit(3).all()
        parts = [f"id={row.id}:{kind_of(getattr(row, attr))}" for row in rows]
        print(f"{label} total={total} nonnull={nonnull} samples=" + ", ".join(parts))
        for row in rows:
            value = getattr(row, attr)
            print(f"  sample id={row.id} value={sample_of(value)}")

    pair = db.query(ChatSession.username, ChatSession.id).join(Message, Message.session_id == ChatSession.id).filter(Message.role == 'assistant', Message.request_id.is_not(None)).distinct().order_by(ChatSession.id).first()
    if not pair:
        pair = db.query(ChatSession.username, ChatSession.id).order_by(ChatSession.id).first()
    if not pair:
        print("history blocker=no chat_sessions found")
    else:
        username, session_id = pair
        try:
            history = get_session_history(db, username, session_id)
            retrieval_items = [item.get("retrieval") for item in history if isinstance(item, dict) and item.get("retrieval") is not None]
            print(f"history username={username} session_id={session_id} messages={len(history)} retrieval_items={len(retrieval_items)} status=ok")
            if history:
                first = history[0]
                print(f"  first_message role={first.get('role')} retrieval_type={type(first.get('retrieval')).__name__ if first.get('retrieval') is not None else 'None'}")
        except Exception as exc:
            print(f"history username={username} session_id={session_id} status=error error={exc!r}")
finally:
    db.close()
