# RAG Core Quality Audit Daily Report — 2026-05-26

## 1. Scope

Inspected the current RAG/chat behavior for:

- `POST /api/v1/chat/stream`
- direct chat vs RAG chat routing
- session and document scoping
- uploaded PDF ingestion
- chunking and metadata quality
- Qdrant/vector retrieval and database fallback
- RAG evidence prompt and guardrails
- related tests and rag-core parity paths

Backend paths inspected:

- `app/api/endpoints/chat.py`
- `app/api/endpoints/knowledge.py`
- `app/services/chat_service.py`
- `app/services/retrieval_service.py`
- `app/services/knowledge_service.py`
- `app/services/vector_store.py`
- `app/services/llm_provider.py`
- `app/crud/crud_knowledge.py`
- `app/models/knowledge_models.py`
- `app/core/config.py`

rag-core paths inspected:

- `rag-core/src/rag_core/parsing/*`
- `rag-core/src/rag_core/chunking/*`
- `rag-core/src/rag_core/indexing/pipeline.py`
- `rag-core/src/rag_core/retrieval/*`
- `rag-core/src/rag_core/vector_store/qdrant_adapter.py`

Related flows inspected:

- Chat stream route -> `handle_chat_stream()` -> `_prepare_chat_turn()` -> optional `search_knowledge()` -> LLM provider stream.
- Knowledge upload route -> `extract_text_from_file()` -> `_execute_indexing()` -> ingestion pipeline -> embeddings -> DB chunks/Qdrant.
- Retrieval route -> query expansion -> query embedding -> Qdrant or DB candidates -> hybrid scoring -> rerank -> prompt context.

No `.env` file was opened or read.

## 2. Executive Summary

Current RAG quality status: usable for simple document QA where one or a few chunks contain the answer. Not good enough for reliable section-level document QA that requires finding a heading, collecting all content until the next heading, preserving order, counting items, and summarizing each item.

No-document chat path: YES, verified. If there is no selected `knowledge_document_id` and no session-bound document row, RAG retrieval is bypassed. The backend still checks for session documents, but it does not call embeddings, Qdrant, database retrieval, or rag-core retrieval.

Document/session isolation: user isolation is good; session isolation has a high-risk edge case. If a session has documents but none are indexed, automatic RAG activates and `search_knowledge()` can switch to `session_scope="global"`, pulling unrelated global documents for the same user.

Main risks found:

- Section-level questions like "Bài tập thực hành số 4 có mấy bài..." are not supported by the retrieval model.
- Session-bound chats can be polluted by global documents when session documents are not indexed.
- PDF page/section metadata is not preserved structurally in the default pipeline.
- Qdrant failures fall back to DB search without strong metadata/analytics visibility.
- Custom chunk size/overlap settings are not honored by the default custom ingestion pipeline.

## 3. Findings

### Finding 1: No-document chat behavior is direct-chat mode

Severity: Low

Evidence from code/logs:

- `_should_use_knowledge_base()` returns false unless `knowledge_document_id` exists or session documents exist: `app/services/chat_service.py:1047-1054`.
- `search_knowledge()` is only called inside `if knowledge_base_active`: `app/services/chat_service.py:1134-1147`.
- Direct-chat prompt branch says no knowledge-base evidence is attached and answers normally: `app/services/chat_service.py:752-767`.
- Added and passed `tests/test_chat_rag_routing.py`.

Impact:

- No-doc chat avoids embedding and retrieval cost.
- The LLM receives system prompt + chat history + latest user message.

Recommended fix:

- Keep the behavior.
- Add a stream endpoint-level regression test with mocked LLM provider to protect it.

### Finding 2: RAG is not called unnecessarily for true no-document sessions

Severity: Low

Evidence from code/logs:

- `_prepare_chat_turn()` always calls `crud_knowledge.list_documents()` to decide the gate, but does not call `search_knowledge()` when the result is empty.
- Targeted test selection passed: `57 passed, 316 deselected`.

Impact:

- Direct chat does not call `embed_query()`, Qdrant, DB retrieval, or rag-core retrieval.

Recommended fix:

- No runtime fix needed.
- Keep a regression test.

### Finding 3: Retrieval is owner-scoped but session scoping is unsafe when session docs are not indexed

Severity: High

Evidence from code/logs:

- Explicit selected doc validates owner and session compatibility: `app/services/chat_service.py:1076-1081`.
- DB retrieval filters `KnowledgeDocument.owner_username`: `app/crud/crud_knowledge.py:180-207`, `229-252`.
- Qdrant search filters `owner_username`: `rag-core/src/rag_core/vector_store/qdrant_adapter.py:278-283`.
- But `search_knowledge()` sets `session_scope="global"` when `document_id is None`, `session_id` is present, and `has_indexed_documents_for_session()` is false: `app/services/retrieval_service.py:170-176`.
- `list_documents()` used by chat gate does not filter status, only owner/session/deleted: `app/crud/crud_knowledge.py:78-105`.

Impact:

- If a PDF is uploaded to a session but indexing is still queued/processing/failed, RAG can use global user documents instead.
- This can produce unrelated answers that look grounded.

Recommended fix:

- Auto-enable session RAG only for indexed session documents.
- Never fall back to global docs from a session-bound chat unless explicitly requested.
- Return "document still indexing / no indexed docs for this session" metadata instead.

### Finding 4: PDF parsing quality is basic and loses structured provenance

Severity: Medium

Evidence from code/logs:

- PDF extraction uses PyMuPDF `get_text("blocks")`, sorts blocks by page coordinate, and inserts `[Page N]` as text: `app/services/knowledge_service.py:216-259`.
- The default custom chunker does not turn `[Page N]` into `page_number` metadata.

Impact:

- Page citations are unreliable.
- Later retrieval cannot expand or cite by page.

Recommended fix:

- Preserve per-page units through chunking.
- Store `page_number` or `page_range` in chunk metadata.

### Finding 5: Chunking quality is insufficient for section-level QA

Severity: High

Evidence from code/logs:

- Default chunking is sentence/newline splitting around 800 chars and 100 char overlap: `rag-core/src/rag_core/chunking/sentence_chunker.py:18-37`.
- The custom pipeline emits chunks but does not detect headings or section boundaries: `rag-core/src/rag_core/chunking/custom_pipeline.py:71-88`.

Impact:

- A section like `Bài thực hành số 4` can be split across multiple independent chunks with no section ID.
- Counting and summarizing every exercise in a section is not reliable.

Recommended fix:

- Add heading-aware chunk metadata and section IDs.
- Add section expansion retrieval.

### Finding 6: Chunk metadata quality is incomplete

Severity: Medium

Evidence from code/logs:

- `IngestionChunk` supports `page_number` and `section_title`: `rag-core/src/rag_core/chunking/base.py:34-39`, `70-73`.
- Custom pipeline normally provides neither.
- Retrieval results do not pass page/section metadata into the prompt: `app/services/retrieval_service.py:277-294`.

Impact:

- Source citations cannot show page or section.
- Even if another pipeline supplies metadata, answer generation does not benefit.

Recommended fix:

- Thread safe metadata fields through retrieval results, prompt evidence headers, and answer citations.

### Finding 7: Qdrant retrieval status is fixed in current code, but deployment should be verified

Severity: Medium

Evidence from code/logs:

- User-provided recent logs showed prior failure: `'QdrantClient' object has no attribute 'search'`.
- Current adapter calls `query_points()`: `rag-core/src/rag_core/vector_store/qdrant_adapter.py:313-320`.
- Regression test asserts `search()` is not called: `tests/test_qdrant_retrieval_query_points.py`.
- Local `qdrant-client` version checked via venv package metadata: `1.14.3`.

Impact:

- Current code should not hit the old `.search` path.
- If the old log persists, the running backend may be using stale code or a different rag-core import.

Recommended fix:

- Verify deployed import path and restart backend.
- Keep the `query_points` regression test.

### Finding 8: top_k / score threshold / reranking behavior is too chunk-centric

Severity: Medium

Evidence from code/logs:

- Default `RETRIEVAL_TOP_K=5`, `RETRIEVAL_MAX_CONTEXT_CHUNKS=6`, `RETRIEVAL_MAX_CONTEXT_TOKENS=4000`: `app/core/config.py:276`, `336-345`.
- Default thresholds: `RETRIEVAL_MIN_SCORE=0.15`, `RETRIEVAL_MIN_LEXICAL_SCORE=0.1`, `RETRIEVAL_LOW_CONFIDENCE_SCORE=0.2`: `app/core/config.py:276-283`, `318-323`.
- Reranker uses only title lexical overlap and chunk-index position boost: `rag-core/src/rag_core/retrieval/reranker.py:19-80`.

Impact:

- Retrieval may return only the best scattered chunks, not all chunks needed for a section.
- Weak scores trigger cautious/insufficient answer behavior.

Recommended fix:

- Add query intent detection for section/count/summarize tasks.
- Dynamically expand context by section or adjacent chunks instead of relying on fixed top-k.

### Finding 9: Section-level questions cannot be answered accurately today

Severity: High

Evidence from code/logs:

- No section boundary parser was found.
- No "fetch chunks from heading until next heading" function was found.
- Prompt only sees top-k evidence blocks, not a complete section.

Impact:

- The specific Vietnamese query can retrieve partial content and produce an insufficient-evidence answer even when the PDF was imported.

Recommended fix:

- Implement section retrieval:
  - detect headings during ingestion;
  - store section key/title/order/page range;
  - resolve section number from query;
  - fetch ordered chunks in that section;
  - pack them as a complete section context.

### Finding 10: Exact quote/exact section retrieval is not supported

Severity: Medium

Evidence from code/logs:

- Search uses vector/lexical chunk scoring, not exact heading lookup or quote extraction.
- Citation snippets are generated by truncating chunk text to 220 chars: `rag-core/src/rag_core/retrieval/evidence.py:20-33`.

Impact:

- Users cannot reliably ask "quote the section" or "summarize all items under section 4".

Recommended fix:

- Add exact heading/section lookup path.
- Preserve exact source spans for citations.

### Finding 11: Database fallback hides vector retrieval failures

Severity: Medium

Evidence from code/logs:

- `search_knowledge()` catches any Qdrant exception, logs a warning, and falls back to database candidate search: `app/services/retrieval_service.py:209-227`.
- Returned retrieval metadata includes `fallback_used`, but that field is currently used for document-scope seed fallback, not Qdrant failure.

Impact:

- Vector search can be broken while answers still return, masking quality regressions and cost changes.

Recommended fix:

- Add retrieval metadata for vector-store attempted/failed/error type.
- Surface fallback rate in admin analytics.

### Finding 12: Configured chunk size/overlap are not honored by the default custom pipeline

Severity: Medium

Evidence from code/logs:

- Config exposes `CHUNK_SIZE` and `CHUNK_OVERLAP`: `app/core/config.py:399-400`.
- Backend factory passes them to rag-core factory.
- rag-core factory ignores them for `custom`: `rag-core/src/rag_core/chunking/factory.py:52-56`.
- `CustomPipeline` calls `chunk_text(text)` without explicit size/overlap: `rag-core/src/rag_core/chunking/custom_pipeline.py:71`.

Impact:

- Tuning chunking through env/config may have no effect under the default pipeline.

Recommended fix:

- Store chunk size/overlap on `CustomPipeline` and pass them into `chunk_text()`.

## 4. Answer to the key question

YES, verified.

If the current chat/session has no imported/attached document and no explicit `knowledge_document_id`, the system behaves like a normal chatbot for knowledge retrieval purposes. It sends the LLM a system prompt, sanitized chat history, and the latest user prompt. It does not call rag-core retrieval, `embed_query()`, Qdrant, or database retrieval.

Why:

- `_should_use_knowledge_base()` returns false when both selected document and session documents are absent.
- `search_knowledge()` is inside the `if knowledge_base_active` block.
- The no-doc prompt branch explicitly tells the assistant to answer normally.
- The added test verifies `search_knowledge.assert_not_called()`.

Nuance:

- The system still queries the DB to list session documents.
- If `use_web_search=True`, Tavily web search can run, but that is separate from rag-core retrieval.
- If a session has document rows that are not indexed, the behavior changes and may incorrectly retrieve global documents.

## 5. Recommended fixes ranked by priority

P0: must fix immediately

- Prevent implicit global retrieval for session-bound chats.
- Only auto-enable RAG for indexed session documents.
- Expose vector-store failure metadata instead of hiding it behind DB fallback.

P1: should fix soon

- Add section-aware ingestion metadata for headings such as `Bài thực hành số N`.
- Add section retrieval that fetches all chunks in a matched section in source order.
- Preserve page/section metadata through retrieval, prompt, sources, and citations.

P2: improvement

- Make `CHUNK_SIZE` and `CHUNK_OVERLAP` effective for custom ingestion.
- Add adjacent chunk expansion around high-confidence hits.
- Add exact heading/BM25 retrieval for numbered sections.
- Dynamically increase context for count/summarize section questions.

P3: optional enhancement

- Add a learned reranker.
- Add PDF table/list item extraction.
- Add dashboard metrics for weak evidence, insufficient evidence, and fallback rates.

## 6. Suggested implementation plan

Minimal safe plan:

1. Change chat RAG gating to use indexed session documents only.
2. Change retrieval scoping so session-bound retrieval never falls back to global automatically.
3. Add metadata fields for Qdrant failure and session no-indexed-doc state.
4. Add tests:
   - no-doc direct chat does not call retrieval;
   - session with only non-indexed docs does not retrieve global docs;
   - session with indexed docs retrieves only that session;
   - selected global document retrieves only that document.
5. Add page/section metadata pass-through.
6. Implement section-aware retrieval as a small additive path before broad vector search.

This avoids a large rewrite and fixes the highest-risk behavior first.

## 7. Tests/validation performed

Virtualenv located:

- Repo root `.venv`: not present.
- Backend `.venv`: present at `C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv`.

Python used:

`C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe`

Commands run:

```powershell
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('qdrant-client'))"
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe -c "import rag_core; print(rag_core.__file__)"
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe -m pytest tests -k "chat or retrieval or knowledge or qdrant"
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe -m pytest tests
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe -m pytest tests --basetemp=.pytest-tmp
```

Results:

- Python executable verified as backend `.venv`.
- `qdrant-client` version: `1.14.3`.
- `rag_core` import path: `C:\Users\Admin\Documents\DominicChatbot\rag-core\src\rag_core\__init__.py`.
- Targeted tests: `57 passed, 316 deselected, 11 warnings`.
- Full suite first run: `367 passed, 6 errors`; errors were pytest temp-directory permission errors under the user temp directory, not code failures.
- Full suite rerun with backend-local `--basetemp=.pytest-tmp`: `373 passed, 33 warnings`.

Added lightweight diagnostic test:

- `tests/test_chat_rag_routing.py` verifies no-document `_prepare_chat_turn()` does not call `search_knowledge()`.

Skipped tests:

- No real external Qdrant/embedding/LLM API calls were made. Existing tests use mocks for those paths.

## 8. Risks and open questions

- Was the reported PDF uploaded with a `session_id`? If not, automatic session RAG will not use it unless selected explicitly.
- Was indexing complete before the section-level question was asked? If not, the current system may have searched global docs.
- Does the PDF contain selectable text, or is it scanned/image-heavy? PyMuPDF text extraction will be weak for scanned PDFs unless image caption/OCR handling is enabled and effective.
- What exact heading text was extracted for practice 4?
- Is the running backend using the current `rag-core/src` path and Qdrant adapter?
- What embedding provider/model/dimensions are active in runtime? Code defaults are local hash/64, but recent logs indicate NVIDIA embeddings are configured externally.
