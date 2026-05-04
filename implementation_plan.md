# 🔍 Dominic Project — Full Audit Report

> **Date**: 2026-05-02 | **Scope**: Frontend (`Dominic/chatbot-ui`) + Backend (`DominicBE`)

---

## 📊 Executive Summary

| Category | Count | Severity |
|---|---|---|
| 🔴 Critical Bugs | 3 | Must fix |
| 🟠 Security Issues | 4 | High priority |
| 🟡 Missing Features | 5 | Medium priority |
| 🔵 Redundancy / Dead Code | 7 | Low-Medium |
| ⚪ Code Quality | 6 | Low |
| 🟢 Architecture Improvements | 4 | Long-term |

### Previous Audit Progress (so sánh với review trước)

| Issue from Previous Audit | Status |
|---|---|
| ❌ Streaming Response missing | ✅ **Fixed** — `POST /chat/stream` SSE endpoint implemented |
| ❌ Rate Limiting missing | ✅ **Fixed** — `FixedWindowRateLimiter` + `DatabaseFixedWindowRateLimiter` |
| 🔴 Hardcoded Admin username | ✅ **Fixed** — `set_user_role` now properly accepts role parameter |
| ❌ Message Pagination missing | ✅ **Fixed** — `skip`, `limit`, `before_id` params added |
| ❌ Input Validation missing | ✅ **Fixed** — Message length, username policy, image validation |
| ❌ API Versioning missing | ✅ **Fixed** — Dual `/api/` + `/api/v1/` mount |
| ❌ Audit Logging missing | ✅ **Fixed** — `AuditLog` model + endpoints |
| ❌ Database Migration Validation | ✅ **Fixed** — Startup validation with `warn`/`strict` modes |
| 🔴 Rename Session disabled | ⚠️ **Still broken** — Still raises `PermissionError` |
| ❌ Unit Tests missing | ⚠️ **Still missing** — 0 backend Python tests |
| ❌ Token Refresh mechanism | ⚠️ **Still missing** |
| ⚠️ Duplicate Endpoints | ⚠️ **Still present** |
| ⚠️ Error Message Leak | ⚠️ **Still present** (28+ `except Exception as e: raise HTTPException(500, str(e))`) |

---

## 🔴 Phase 1: Critical Bugs

### BUG-01: `rename_session` is permanently disabled

**Location**: [chat_service.py:286-287](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/services/chat_service.py#L286-L287)

```python
def rename_session(db: Session, username: str, session_id: int, title: str) -> dict:
    raise PermissionError("Manual chat renaming is disabled.")
```

- Backend endpoint `PATCH /sessions/{session_id}` exists and is mounted
- Frontend exports `renameSession()` in [chatApi.js:248](file:///c:/Users/Admin/Documents/GitHub/Dominic/chatbot-ui/src/service/chatApi.js#L248)
- Any attempt to call this returns 403 with no workaround
- **Impact**: Dead endpoint wasting API surface, confusing developers

> [!CAUTION]
> Either implement the rename logic properly OR remove the endpoint + frontend function entirely.

---

### BUG-02: Frontend ESLint errors (9 errors, build passes but code quality)

**Location**: Multiple frontend files

| File | Error | Impact |
|---|---|---|
| `App.jsx:1` | `motion` imported but unused | Dead import |
| `ChatInput.jsx:1` | `motion` imported but unused | Dead import |
| `ChatInput.jsx:181` | React Compiler memoization failure | Performance degradation |
| `Login.jsx:1` | `motion` imported but unused | Dead import |
| `Login.jsx:149` | `setState` called synchronously in effect | Cascading re-renders |
| `MessageBubble.jsx:255` | `_node` defined but unused | Dead variable |
| `Sidebar.jsx:1` | `motion` imported but unused | Dead import |
| `Sidebar.jsx:122` | `accountMenuOpen` assigned but never used | Dead variable |

> [!WARNING]
> The `Login.jsx:149` issue (`setText` inside effect body) can cause cascading renders hurting performance on the login screen. The `ChatInput.jsx:181` memoization failure means React Compiler skips optimizing the entire ChatInput component.

---

### BUG-03: `datetime.utcnow()` deprecated — 5 usages remain

**Locations**:
- [crud_chat.py:147](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/crud/crud_chat.py#L147)
- [crud_chat.py:277](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/crud/crud_chat.py#L277)
- [crud_chat.py:290](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/crud/crud_chat.py#L290)
- [crud_auth.py:126](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/crud/crud_auth.py#L126)
- [crud_auth.py:141](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/crud/crud_auth.py#L141)

`datetime.utcnow()` was deprecated in Python 3.12. The codebase already uses `datetime.now(UTC)` in [security.py:100](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/core/security.py#L100), so this is an inconsistency that could cause timezone bugs when comparing timestamps across modules.

---

## 🟠 Phase 2: Security Issues

### SEC-01: Internal error messages leaked to clients (28+ instances)

Every API endpoint catches `Exception` and returns `str(e)` to the client:

```python
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

This exposes:
- Database error messages (table names, column names, connection strings)
- File system paths
- Stack trace information
- Internal service names

**Count**: 28+ instances across `auth.py`, `chat.py`, `knowledge.py`

> [!CAUTION]
> In production, this is an information disclosure vulnerability. Should log full errors internally and return generic user-facing messages.

---

### SEC-02: `.env` file committed to Git with real API keys

**Location**: [.env](file:///c:/Users/Admin/Documents/GitHub/DominicBE/.env)

Contains:
- `TAVILY_API_KEY=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`
- `GITHUB_COPILOT_API_KEY=sk-97d60c79263819f8-geg3tf-2f87be70`
- `OBJECT_STORAGE_SECRET_KEY=minioadmin123`
- `DATABASE_URL` with password

> [!CAUTION]
> `.env` is NOT in `.gitignore`. Real API keys are committed to version control. These should be rotated immediately and `.env` added to `.gitignore`.

---

### SEC-03: CORS regex allows arbitrary Azure Static Web Apps subdomains

**Location**: [main.py:194](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/main.py#L194)

```python
allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.azurestaticapps\.net$"
```

Any app deployed on Azure Static Web Apps (free tier) can make authenticated requests to your backend.

---

### SEC-04: No JWT token revocation mechanism

- Token TTL = 7 days (`AUTH_ACCESS_TOKEN_EXPIRE_MINUTES=10080`)
- No refresh token flow
- No token blacklist
- Logout only clears client-side storage — token remains valid server-side
- A leaked token cannot be revoked

---

## 🟡 Phase 3: Missing Features

### MISS-01: Zero backend Python unit/integration tests

**Status**: No `tests/` directory in `DominicBE`. Only:
- 2 throwaway scripts in `scripts/` (`test_login.py`, `test_image_processor.py`)
- Frontend has 3 Playwright E2E specs (with mocked API)

No testing for:
- CRUD operations
- Authentication flows
- RAG pipeline logic
- Rate limiting behavior
- Schema validation
- Answer guardrails

---

### MISS-02: No database pool sizing configuration

**Location**: [database.py:29-35](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/core/database.py#L29-L35)

```python
engine = create_engine(
    settings.sqlalchemy_database_url,
    pool_pre_ping=True,
    pool_recycle=settings.db_pool_recycle,
    pool_timeout=settings.db_pool_timeout,
    connect_args=connect_args,
    # Missing: pool_size, max_overflow
)
```

Uses SQLAlchemy defaults (`pool_size=5`, `max_overflow=10`). Under concurrent load with rate limiting hitting the DB, this could exhaust connections quickly.

---

### MISS-03: Knowledge duplicate document detection not enforced

- `KnowledgeDocument` model has `checksum` column with an index
- `knowledge_service.py` computes checksums during ingestion
- But **no check for duplicate checksums** before creating a new document
- Uploading the same file twice creates two separate document records with duplicate chunks

---

### MISS-04: Frontend bundle too large — no code splitting

```
dist/assets/index-CTPrurP0.js   604.79 kB │ gzip: 186.42 kB
```

Vite warns: "Some chunks are larger than 500 kB after minification." The entire app is a single monolithic JS bundle. No `React.lazy()` or dynamic `import()` for:
- `AdminPanel` (admin-only)
- `KnowledgePanel` (secondary view)
- `Login` (pre-auth only)

---

### MISS-05: No rate limiting on chat endpoints

Rate limits are configured for:
- `auth.register`, `auth.login`, `auth.reset_password`
- `knowledge.upload`, `knowledge.ingest`

But **NOT** for:
- `POST /chat/` (synchronous chat)
- `POST /chat/stream` (streaming chat — most expensive endpoint)

A malicious user could exhaust LLM API quota rapidly.

---

## 🔵 Phase 4: Redundancy & Dead Code

### RED-01: Duplicate "me" vs "{username}" API endpoints

| "Me" Endpoint | "{username}" Endpoint | Redundant? |
|---|---|---|
| `GET /usage/me` | `GET /usage/{username}` | ⚠️ Same logic |
| `GET /sessions` | `GET /sessions/{username}` | ⚠️ Same logic |
| `GET /sessions/{id}/messages` | `GET /sessions/{username}/{id}/messages` | ⚠️ Same logic |

`_assert_same_user()` only allows users to access their own data, so the `{username}` variants add no capability. They just double the API surface and maintenance burden.

---

### RED-02: 7 temporary files in backend root

```
.tmp_backfill_validation.py
.tmp_health_direct.py
.tmp_health_exact.py
.tmp_health_validation.py
.tmp_json_validation.py
.tmp_psycopg_output.txt
.tmp_search.ps1
.tmp_validation_run.py
migration_mysql_to_postgres_run.log
migration_run.log
```

10 scratch/debug files sitting in the project root. Should be gitignored or deleted.

---

### RED-03: Prompt caching is a no-op

**Location**: `llm_provider.py` — `_apply_prompt_caching()` function

Config settings `LLM_PROMPT_CACHING_ENABLED` and `LLM_PROMPT_CACHING_MIN_CHARS` exist but the function just returns its inputs unchanged. Dead config polluting the settings namespace.

---

### RED-04: Duplicate `CitationSource` schema classes

- [chat_schemas.py:205](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/schemas/chat_schemas.py#L205) — `CitationSource` with web-specific fields
- [knowledge_schemas.py:146](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/schemas/knowledge_schemas.py#L146) — `CitationSource` with knowledge-specific fields

Two different classes with the same name in different modules. Easy to confuse when importing.

---

### RED-05: `renameSession` exported but never called in frontend

[chatApi.js:248](file:///c:/Users/Admin/Documents/GitHub/Dominic/chatbot-ui/src/service/chatApi.js#L248) exports `renameSession()` but `App.jsx` never imports or uses it. Dead export.

---

### RED-06: `image_payload_json` column is overloaded

[chat_models.py:30](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/models/chat_models.py#L30) — Single `Text` column stores:
- Images (base64 URIs)
- Documents (attachment metadata)
- Sources (web citations)
- Assistant metadata (model name, reasoning effort)

All packed into one JSON blob. Difficult to query, index, or extend independently.

---

### RED-07: Unused `motion` import in 4 frontend components

`AnimatePresence` is used but the `motion` named export is imported and unused in:
- `App.jsx` (uses `motion.div` but import pattern confuses ESLint)
- `ChatInput.jsx`
- `Login.jsx`
- `Sidebar.jsx`

---

## ⚪ Phase 5: Code Quality Issues

### CQ-01: `App.jsx` is a 1571-line monolith

The main application component contains:
- Authentication state management
- Chat session orchestration
- Knowledge document management
- Admin panel data loading
- Image/document caching
- Stream handling
- Error handling
- All event handlers

Should be decomposed using custom hooks:
- `useAuth()` — authentication & session
- `useChatSessions()` — session CRUD
- `useChatMessages()` — message loading & streaming
- `useKnowledge()` — document management
- `useAdmin()` — admin panel data

---

### CQ-02: `chat_service.py` is a 1465-line monolith

Backend service with 10+ concerns:
- Session management
- Message history building
- Retrieval orchestration
- Answer guardrails & evidence grading
- Web source formatting
- Auto-titling logic
- Summary compression
- Token estimation
- System prompt composition
- LLM call orchestration

---

### CQ-03: `config.py` default DB port is MySQL (`3306`) but project uses PostgreSQL

```python
db_port: int = Field(default=3306, alias="DB_PORT", ge=1, le=65535)
```

The default port is MySQL's `3306`, but the project exclusively uses PostgreSQL (port `5432`). The `sqlalchemy_database_url` property also generates MySQL URLs by default (`mysql+pymysql://`).

While overridden by `DATABASE_URL` env var, the defaults are misleading and would fail for new developers.

---

### CQ-04: Frontend error messages mix Vietnamese and ASCII-only Vietnamese

Some messages use proper Unicode Vietnamese:
```js
"Đang tải đoạn chat của bạn"
"Phiên đăng nhập đã hết hạn."
```

Others use ASCII-only (no diacritics):
```js
"Co loi khi goi API chat."
"Khong ket noi duoc backend"
"Dang nhap that bai."
```

Inconsistent UX for Vietnamese users.

---

### CQ-05: `claude-opus-4.6` marked as unavailable in frontend config

[uiConfig.js:28-30](file:///c:/Users/Admin/Documents/GitHub/Dominic/chatbot-ui/src/config/uiConfig.js#L28-L30):

```js
export const CHAT_MODEL_AVAILABILITY = {
    "claude-opus-4.6": false,
};
```

The model is in `SUPPORTED_CHAT_MODELS` but explicitly marked unavailable. Users see it in the model list but can't select it. Should either enable it or remove it from the list entirely.

---

### CQ-06: MySQL driver still in requirements despite PostgreSQL-only usage

[requirements.txt:4](file:///c:/Users/Admin/Documents/GitHub/DominicBE/requirements.txt#L4):

```
pymysql==1.1.0
```

The project uses PostgreSQL exclusively (`psycopg`). `pymysql` is an unnecessary dependency.

---

## 🟢 Phase 6: Architecture Improvements

### ARCH-01: Synchronous DB operations in async context

`upload_document` endpoint is `async def` but all DB operations are synchronous SQLAlchemy. This blocks the event loop during database queries. Either:
- Make the endpoint synchronous (`def` instead of `async def`)
- Use SQLAlchemy async engine with `asyncpg`

---

### ARCH-02: No OpenAPI documentation enhancement

- Most endpoints lack `description`, `summary`, or `response_description`
- No example responses in schema models
- No custom `/docs` page branding
- API consumers have minimal guidance

---

### ARCH-03: Background job recovery uses thread at startup

[main.py:93-118](file:///c:/Users/Admin/Documents/GitHub/DominicBE/app/main.py#L93-L118) — Ingestion recovery runs in a daemon `Thread`. If the process restarts frequently, recovery jobs might overlap. Consider using a proper task queue (Celery, ARQ) for production.

---

### ARCH-04: Rate limiting hits database on every request

When `RATE_LIMIT_STORE_BACKEND=database` (default), every rate-limited request:
1. Opens a DB session
2. Runs a `SELECT ... FOR UPDATE` query
3. Inserts or updates a row
4. Commits

This adds ~5-20ms latency to every request. For high-traffic endpoints, consider Redis-backed rate limiting.

---

## 📋 Recommended Fix Priority

### Immediate (Sprint 1)
1. **SEC-02**: Remove `.env` from Git, rotate all API keys
2. **BUG-02**: Fix ESLint errors (especially Login.jsx cascading renders)
3. **SEC-01**: Replace `str(e)` with generic error messages in all endpoints
4. **BUG-03**: Replace `datetime.utcnow()` with `datetime.now(UTC)`

### Short-term (Sprint 2)
5. **BUG-01**: Either implement `rename_session` or remove the dead endpoint
6. **MISS-05**: Add rate limiting to chat endpoints
7. **RED-02**: Clean up temporary files
8. **CQ-04**: Standardize Vietnamese error messages

### Medium-term (Sprint 3-4)
9. **MISS-01**: Add pytest test suite for critical paths
10. **MISS-04**: Add code splitting for AdminPanel, KnowledgePanel, Login
11. **CQ-01**: Decompose `App.jsx` into custom hooks
12. **CQ-03**: Fix default DB config to PostgreSQL

### Long-term (Backlog)
13. **SEC-04**: Implement refresh token mechanism
14. **MISS-02**: Configure DB pool sizing
15. **MISS-03**: Add duplicate document detection
16. **ARCH-01**: Resolve sync/async mismatch
17. **RED-01**: Deprecate redundant `{username}` endpoints
