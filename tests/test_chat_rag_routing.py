from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


os.environ["DEBUG"] = "false"
os.environ["ENVIRONMENT"] = "local"
os.environ["AUTH_SECRET_KEY"] = "test-secret-key-for-unit-tests-only"


def test_prepare_chat_turn_without_documents_bypasses_rag_retrieval():
    from app.services import chat_service

    db = MagicMock()
    user = SimpleNamespace(username="alice", max_tokens_per_day=10000)
    session = SimpleNamespace(id=42, username="alice", title="Direct chat")
    user_msg = SimpleNamespace(id=1001)
    model_selection = SimpleNamespace(context_window=4096, max_output_tokens=512)
    registry = MagicMock()
    registry.select_model.return_value = model_selection

    with patch.object(chat_service.crud_chat, "get_user_by_username", return_value=user), \
         patch.object(chat_service.crud_chat, "get_chat_session", return_value=session), \
         patch.object(chat_service.crud_chat, "count_session_messages", return_value=0), \
         patch.object(chat_service.crud_chat, "get_rolling_token_usage", return_value={"total_tokens": 0}), \
         patch.object(chat_service.crud_chat, "create_message", return_value=user_msg), \
         patch.object(chat_service.crud_knowledge, "list_documents", return_value=[]), \
         patch.object(chat_service, "_build_hybrid_context", return_value=(None, [])), \
         patch.object(chat_service.llm_provider, "get_provider_registry", return_value=registry), \
         patch.object(chat_service, "search_knowledge") as search_knowledge:
        prepared = chat_service._prepare_chat_turn(
            db,
            "alice",
            42,
            "hello",
            knowledge_document_id=None,
            use_web_search=False,
        )

    search_knowledge.assert_not_called()
    assert prepared.knowledge_base_active is False
    assert prepared.retrieval_result["rag_mode"] == "direct_chat"
    assert prepared.retrieval_result["retrieval_scope"] == "none"
    assert prepared.retrieval_result["vector_store_attempted"] is False
    assert prepared.request_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert "No knowledge-base evidence is attached" in prepared.request_kwargs["system"]
