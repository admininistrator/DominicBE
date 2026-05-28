# RAG Core Quality Update Daily Report — 2026-05-27

> **Author**: Global System Auditor (Reviewer — Final End-to-End Review)
> **Feature**: `2026-05-26-rag-retrieval-quality`
> **Status**: APPROVED
> **Date**: 2026-05-28T06:48:00Z

---

## 1. Scope

This daily report summarizes the completed RAG Retrieval Quality feature, implemented across four phase groups spanning two repositories ([`DominicBE`](../DominicBE/) and [`rag-core`](../rag-core/)), and validated with a final end-to-end review.

The feature addresses six findings from the 2026-05-26 RAG quality audit:
1. Preserve correct no-document direct chat behavior
2. Fix session-to-global fallback (high-risk safety bug)
3. Build metadata foundation (page, span, section)
4. Implement section-aware retrieval MVP
5. Tune chunking and retrieval parameters
6. Provide backfill/re-index and rollout documentation

---

## 2. Summary of Completed Work

### Phase Group 1 — Session Safety and Retrieval Observability
- **Implemented**: Indexed-only RAG gate, no session-to-global fallback, 10-field retrieval metadata contract, safe vector-store failure observability, 5 regression tests
- **Reviewed**: ✅ APPROVED
- **Files**: [`crud_knowledge.py`](../DominicBE/app/crud/crud_knowledge.py), [`retrieval_service.py`](../DominicBE/app/services/retrieval_service.py), [`chat_service.py`](../DominicBE/app/services/chat_service.py), [`test_session_scope_safety.py`](../DominicBE/tests/test_session_scope_safety.py)

### Phase Group 2 — Metadata Foundation + Section-Aware Retrieval
- **Implemented**: Pure heading detector, monotonic span mapper, PDF page sentinel mapping, section metadata assignment, metadata threading, pure section intent/matching, backend section lookup, section retrieval MVP before vector search, synthetic `Bài thực hành số 4` smoke/eval
- **Reviewed**: ✅ APPROVED (with 3 Fixer adjustments)
- **Files**: 11 production files + 11 test files across both repos

### Phase Group 3 — Retrieval Tuning
- **Implemented**: Configurable chunk size/overlap pass-through, adjacent chunk expansion for standard retrieval, pure heading-like matching optimization, capped dynamic context expansion, 86 new/updated tests
- **Reviewed**: ✅ APPROVED
- **Files**: 7 production files + 4 test files

### Phase Group 4 — Documentation, Backfill, and Rollout
- **Implemented**: Backend README docs, Pydantic schema updates for metadata passthrough, smoke/golden set with section coverage, dry-run-first backfill script, rollout notes with optional PostgreSQL index guidance
- **Reviewed**: ✅ APPROVED
- **Files**: 8 files (docs, schemas, scripts, tests)

### Fixer Passes
- **Phase Group 1 Fixer**: Test-only session seed fix for DB-fallback regression — 1 file changed
- **Phase Group 2 Fixer**: `rag-core/pyproject.toml` pythonpath, `retrieval/__init__.py` duplicate `__all__`, `custom_pipeline.py` chunker version reversion — 3 files changed

---

## 3. Current Tech State

### All Phase Groups: APPROVED
All four phase groups have been implemented, tested, fixed (where needed), and individually reviewed with APPROVED decisions. The final end-to-end review confirms no regressions across phase boundaries.

### Validation Evidence
| Suite | Result | Exit Code |
|---|---|---|
| Full backend (`tests --basetemp=.pytest-tmp -v`) | `390 passed, 33 warnings` | 0 |
| rag-core (`rag-core\src\rag_core\tests -v`) | `150 passed` | 0 |
| Knowledge smoke (`knowledge_smoke_test.py`) | `KNOWLEDGE_API_SMOKE_OK` | 0 |
| RAG chat smoke (`rag_chat_smoke_test.py`) | `RAG_CHAT_SMOKE_OK` | 0 |
| Backfill dry-run (`backfill_section_metadata.py --dry-run --limit 1`) | `selected_documents=1 needs_reindex=1 reindexed=0` | 0 |

All commands used the required backend venv: `DominicBE\.venv\Scripts\python.exe` with explicit `PYTHONPATH`.

---

## 4. Retrieval/RAG Architecture After Update

```
Chat request flow:
┌─────────────────────────────────────────────────────────────────┐
│ No documents attached? → direct_chat                            │
│   rag_mode=direct_chat, retrieval_scope=none                    │
│   No embed_query, Qdrant, DB retrieval, or rag-core calls       │
├─────────────────────────────────────────────────────────────────┤
│ Documents but none indexed? → indexing_pending                  │
│   rag_mode=indexing_pending, fallback_reason=no_indexed_...     │
│   No retrieval calls, no global fallback                        │
├─────────────────────────────────────────────────────────────────┤
│ Indexed documents exist?                                        │
│   ├── High-confidence section query? → section_rag              │
│   │     Exact section_key lookup, ordered matched chunks        │
│   │     Skips vector search, embedding, adjacent expansion      │
│   └── Standard retrieval → vector/hybrid                        │
│         ├── Count/list/summarize? → capped dynamic expansion    │
│         └── High-confidence hits? → adjacent chunk expansion     │
│   All paths carry metadata contract + provenance when available │
└─────────────────────────────────────────────────────────────────┘
```

### Retrieval Modes Summary
| `rag_mode` | Scope | Vector Search | Embedding |
|---|---|---|---|
| `direct_chat` | none | No | No |
| `indexing_pending` | session | No | No |
| `session_rag` | session | Yes | Yes |
| `document_rag` | document | Yes | Yes |
| `section_rag` | original | No | No |

---

## 5. Safety and Isolation Improvements

### Session-to-Global Fallback Removed
**Before**: Session-bound chat with non-indexed session documents would silently fall back to global document retrieval, returning answers from unrelated documents.

**After**: Session-bound chats with non-indexed documents return `rag_mode=indexing_pending` with `fallback_reason=no_indexed_session_documents`. No retrieval calls are made. No global documents are consulted.

**Implementation**:
- [`has_indexed_documents_for_session()`](../DominicBE/app/crud/crud_knowledge.py) — efficient existence check with owner, deleted_at, status="indexed", session_id filters
- [`_resolve_retrieval_scope()`](../DominicBE/app/services/retrieval_service.py) — session scope never resolves to `global`
- [`search_knowledge()`](../DominicBE/app/services/retrieval_service.py) — early return without retrieval when no indexed session docs exist
- [`_should_use_knowledge_base()`](../DominicBE/app/services/chat_service.py) — indexed-only RAG activation gate

### Vector-Store Failure Observability
**Before**: Qdrant failures triggered silent DB fallback with no indication to operators.

**After**: DB fallback still works, but response metadata now records `vector_store_attempted=true`, `vector_store_failed=true`, `vector_store_error_type=<ExceptionClassName>` (class name only — no raw messages, URLs, keys, or connection strings), and `fallback_reason=vector_store_failure`.

### Access-Control Preservation
All new code paths preserve existing filters:
- Owner filters in all document/chunk queries
- Session isolation (`session_scope="session"`, `session_id` matching)
- Soft-delete exclusion (`deleted_at.is_(None)`)
- Document status filtering (`status="indexed"`)
- Cross-owner and cross-session isolation explicitly tested

### Metadata Contract Consistency
All 10 fields from the plan are consistently used across [`RETRIEVAL_METADATA_DEFAULTS`](../DominicBE/app/services/retrieval_service.py), [`build_retrieval_metadata_contract()`](../DominicBE/app/services/retrieval_service.py), [`_empty_retrieval_result()`](../DominicBE/app/services/retrieval_service.py), [`search_knowledge()`](../DominicBE/app/services/retrieval_service.py) success path, [`_build_retrieval_payload()`](../DominicBE/app/services/chat_service.py), and [`_build_retrieval_by_request()`](../DominicBE/app/services/chat_service.py).

---

## 6. Metadata and Section Retrieval Improvements

### Page/Section/Span Metadata
Newly ingested or re-indexed chunks now carry:
- `char_start`, `char_end` — source character offsets via monotonic cursor mapping ([`span_mapper.py`](../rag-core/src/rag_core/chunking/span_mapper.py))
- `page_number`, `page_range` — PDF page provenance via exact `<<PAGE:N>>` sentinel mapping ([`page_markers.py`](../rag-core/src/rag_core/chunking/page_markers.py))
- `section_key`, `section_title`, `section_level`, `section_order` — section assignment via nearest-preceding-heading binary search ([`heading_detector.py`](../rag-core/src/rag_core/chunking/heading_detector.py))
- `chunker_version` — `"custom-sentence-v1"`

All metadata stored in existing `metadata_json` column — no schema change.

### Section-Aware Retrieval MVP
High-confidence section queries (e.g., `Bài thực hành số 4 có mấy bài, tóm tắt từng bài`) follow a dedicated path:

1. **Intent detection** (rag-core): Vietnamese/English keyword patterns + regex for academic sections
2. **Section matching** (rag-core): Exact, substring, and Jaccard token similarity against available sections supplied by backend. Confidence threshold: 0.82
3. **Available sections** (backend): `list_available_sections()` with all access-control filters
4. **Ordered chunk fetch** (backend): `find_chunks_by_section_key()` sorted by `(document_id, char_start, chunk_index)`
5. **Result**: `rag_mode=section_rag`, ordered chunks from only the matched section, no vector search or embedding

### rag-core/Backend Boundary
- **rag-core**: Pure logic only — `heading_detector.py`, `page_markers.py`, `span_mapper.py`, `section_retrieval.py`. Zero backend/DB/SQLAlchemy imports.
- **DominicBE**: All DB queries, access-control filters, session management, PDF extraction, retrieval orchestration, metadata passthrough.

### Chunking and Tuning
- **Configurable chunk size/overlap**: `CustomPipeline` accepts optional `chunk_size`/`chunk_overlap`; defaults preserve 800/100 behavior
- **Adjacent chunk expansion**: Standard vector/hybrid retrieval can include neighbor chunks (±1), capped and filtered. Skipped for `section_rag`. Section-boundary guard prevents cross-section mixing when anchors carry metadata.
- **Dynamic context expansion**: Count/list/summarize queries expand `top_k` (capped) when section retrieval fails. Non-expansive queries keep defaults.

---

## 7. Tests and Validation

### Final End-to-End Validation (Executed 2026-05-28T06:45Z)

**Full backend suite**:
```
390 passed, 33 warnings in 50.05s — exit code 0
```
Command: `cd DominicBE && set PYTHONPATH=...rag-core\src;...DominicBE && .venv\Scripts\python.exe -m pytest tests --basetemp=.pytest-tmp -v`

**rag-core suite**:
```
150 passed in 0.88s — exit code 0
```
Command: `cd DominicChatbot && set PYTHONPATH=...rag-core\src && DominicBE\.venv\Scripts\python.exe -m pytest rag-core\src\rag_core\tests -v`

### Smoke Validation
- `KNOWLEDGE_API_SMOKE_OK` — verifies synthetic `Bài thực hành số 4` section retrieval
- `RAG_CHAT_SMOKE_OK` — verifies section-level grounded chat
- Backfill dry-run: `needs_reindex=1, reindexed=0, errors=0` — confirms detection without modification

### Test Properties
- All tests deterministic — no Qdrant, network, `.env`, or production data
- Backend venv used exclusively
- Explicit `PYTHONPATH` ensures correct rag-core import path
- SQLite in-memory/tempfile for all DB-backed tests
- Cross-owner and cross-session isolation explicitly proven

### Phase-Level Test Coverage
| Phase | New/Updated Tests | Total Suite |
|---|---|---|
| PH1 | 5 new + 1 updated | 378 → 378 (after fixer: 378) |
| PH2 | 9 rag-core + 3 backend | 378 → 381 |
| PH3 | 1 rag-core + 4 backend | 381 → 388 |
| PH4 | 2 backend + smoke/golden | 388 → 390 |

---

## 8. Remaining Risks

### Metadata Availability
- **Existing documents lack metadata**: Section/page/span metadata only exists on newly ingested or re-indexed documents. Old documents require backfill via `backfill_section_metadata.py`. Until backfill, section queries against old documents fall back to standard retrieval with `fallback_reason=low_section_confidence`.

### Section Retrieval Limitations
- Conservative heading detection may miss some valid headings → safe fallback to standard retrieval
- Very long sections capped at `top_k` → may truncate context
- MVP Python-side section filtering → large deployments may need PostgreSQL metadata index

### Tuning Considerations
- Adjacent expansion sorts expanded context by document/chunk order → anchors may move from rank 1
- Dynamic context expansion uses keyword-based detection → potential false negatives for uncommon phrasing
- Section-boundary guard in adjacent expansion only applies when anchors carry `section_key` → old chunks may get cross-section neighbors

### Operational
- `backfill_section_metadata.py --apply` not tested against production data
- Stale `rag_core/` directory at repo root may shadow editable install without `PYTHONPATH`
- Smoke commands require safe env overrides in shells with `DEBUG=release`

### Frontend
- Direct-chat additive retrieval payload (not `null`) needs frontend compatibility verification
- Streaming final payload metadata shape should be confirmed matching sync chat

---

## 9. Next Recommendations

### Immediate
1. Verify frontend compatibility with additive direct-chat retrieval payload
2. Confirm streaming final payload metadata shape consistency

### Short-Term
3. Run `backfill_section_metadata.py --dry-run` against production data
4. Batch-apply re-index for critical documents
5. Create PostgreSQL `metadata_json->>'section_key'` index if section retrieval becomes hot

### Medium-Term
6. Monitor section retrieval confidence rates; expand heading patterns as needed
7. Consider section-specific `top_k` for very long sections
8. Add configurable adjacent expansion ordering strategies
9. Evaluate BM25/hybrid retrieval as complement to vector search
10. Remove or rename stale `DominicChatbot/rag_core/` directory

---

## 10. Final Reviewer Decision

### Decision: ✅ APPROVED

The complete RAG Retrieval Quality feature (all four phase groups) passes final end-to-end review with zero blocking issues.

### Evidence
- Full backend suite: **390 passed, 33 warnings** — clean
- rag-core suite: **150 passed** — clean
- Knowledge smoke: `KNOWLEDGE_API_SMOKE_OK`
- RAG chat smoke: `RAG_CHAT_SMOKE_OK`
- Backfill dry-run: safe, no modifications
- All four phase groups individually APPROVED by Tester and Reviewer
- Backend venv used exclusively; no `.env` read; no secrets exposed
- rag-core/backend boundary strictly maintained
- All access-control, session, and document isolation preserved

### Key Achievements
1. ✅ Session safety bug fixed — no more silent global fallback
2. ✅ Retrieval observability — 10-field metadata contract on all paths
3. ✅ Section-aware retrieval — MVP resolves section queries without vector search
4. ✅ Metadata foundation — page/section/span metadata for newly ingested docs
5. ✅ Retrieval tuning — configurable chunking, adjacent expansion, dynamic context
6. ✅ Operational readiness — backfill script, smoke/golden coverage, rollout docs
7. ✅ Zero secret leakage — vector error types are class names only
8. ✅ 390 backend + 150 rag-core tests passing
9. ✅ Clean rag-core/backend boundary with zero DB imports in rag-core

### Remaining Work (Non-Blocking)
- Frontend compatibility verification (additive direct-chat payload)
- Production backfill/re-index for existing documents
- Optional PostgreSQL metadata index for hot section retrieval paths
- Stale `rag_core/` directory cleanup
