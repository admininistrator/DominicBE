# DominicBE

Backend FastAPI cho Dominic. Ở thời điểm hiện tại repo này đã có knowledge ingestion, retrieval và grounded chat; `RAG_UPGRADE_PLAN.md` nên được xem là tài liệu kế hoạch lịch sử, không còn phản ánh đầy đủ trạng thái thực tế của code.

## Trạng thái hiện tại (verified 2026-04-25)

### Các kiểm tra đã chạy trong workspace này

- `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/auth_smoke_test.py` -> pass
- `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/knowledge_smoke_test.py` -> pass
- `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/rag_chat_smoke_test.py` -> pass
- `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/rag_eval_smoke_test.py` -> pass
- `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/test_image_processor.py` -> pass
- `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/test_ocr_injection.py` -> pass
- session-scoped knowledge API check bằng `TestClient` + SQLite in-memory -> pass
- frontend `npm run build` tại `Dominic/chatbot-ui` -> pass
- frontend `npm run lint` tại `Dominic/chatbot-ui` -> fail (14 lỗi; lỗi đầu tiên là biến `activeSessionId` chưa dùng trong `src/components/ChatInput/ChatInput.jsx`)

### Khả năng backend hiện có

#### Phase 1 - Foundation hardening

- auth với register, login, access + refresh token, `/api/v1/auth/me`, `/api/v1/auth/refresh`, `/api/v1/auth/logout`
- versioned API path đã có tại `/api/v1/*`; các route cũ `/api/*` vẫn được giữ làm legacy alias để tránh break client hiện tại
- password hashing, đổi mật khẩu, logout revoke token theo version, admin-issued reset token, public reset-password flow
- quản lý chat session, thống kê usage theo rolling window, conversation summary memory
- history messages hỗ trợ phân trang tùy chọn qua `skip`, `limit`, `before_id`; metadata phân trang trả qua các header `X-Message-Pagination-*`
- input validation đã siết cho username an toàn, chat message tối đa 12000 ký tự, và chat images phải là `data:image/...;base64,...` hợp lệ với media type khớp payload
- migration Alembic, debug endpoint bị khóa mặc định, phân quyền admin/user

#### Phase 2 - Knowledge ingestion MVP

- ingest text và upload file qua `/api/knowledge/documents/ingest` và `/api/knowledge/documents/upload`
- tạo document, chunks, ingestion jobs, reindex, soft delete
- trích xuất/chunk/index tài liệu và xem lại chunks/jobs qua API
- knowledge document có thể gắn với `session_id` để dùng riêng cho một đoạn chat

#### Phase 3 - Retrieval integration

- search knowledge qua `/api/knowledge/search`
- chat response trả `reply`, `usage`, `request_id`, `sources`, `retrieval`
- grounded chat theo tài liệu đã chọn, có fallback khi thiếu bằng chứng
- có thể bật Tavily web search theo từng câu chat khi cấu hình `WEB_SEARCH_ENABLED=true` và `TAVILY_API_KEY` trong `.env`

#### Phase 4 - Frontend RAG UX

- frontend build pass và đã có UI cho knowledge panel, source drawer, retrieval badge
- knowledge có thể import trực tiếp từ ô chat và nhóm theo chat session / global document
- chưa có browser E2E test trong repo; lint frontend hiện chưa sạch

#### Phase 5 - RAG quality improvements

- query expansion, hybrid lexical + semantic scoring, reranking, context packing
- answer guardrails với các policy `grounded`, `cautious_general`, `insufficient_evidence`
- khi evidence yếu, câu trả lời bị hạ mức chắc chắn hoặc trả về thông báo thiếu bằng chứng

#### Phase 6 - Production readiness

- retrieval analytics, audit logs, cost metrics endpoint, soft delete
- admin hard delete đã dọn đồng bộ Postgres metadata, MinIO artifacts và Qdrant points
- `/metrics` endpoint cho Prometheus-style HTTP metrics và `X-Request-ID` header cho request tracing cơ bản
- async indexing qua FastAPI `BackgroundTasks`
- background worker tách rời và live production deployment path chưa được xác thực trong lần rà soát này

### Giới hạn hiện tại

- embedding hiện là local/hash embedding cho MVP, chưa phải semantic embedding production-grade
- backend hiện đã có abstraction cho `DATABASE_URL`, object storage và Qdrant, nhưng chưa được xác thực end-to-end với một cụm Postgres + MinIO/S3 + Qdrant thật trong workspace này
- live provider connection với Anthropic/LiteLLM không được kiểm tra trong workspace này vì phụ thuộc API key/env thật
- frontend chạy được và build được, nhưng vẫn còn debt lint/code cleanup

### Ghi chú môi trường

- để chạy được trên Python 3.13 với code hiện tại, `requirements.txt` cần `sqlalchemy>=2.0.38,<2.1`
- với FastAPI/Starlette hiện có, `httpx` cần pin `>=0.27,<0.28` để `TestClient` hoạt động ổn định

### Storage architecture đã sẵn sàng trong code

## Remote MCP (Model Context Protocol) Integration

### Overview

The backend has integrated **Remote MCP** support, enabling the backend to act as an MCP client and invoke remote MCP tool servers. The first integration is **Excalidraw** (`https://mcp.excalidraw.com`), but the architecture supports N MCP remote servers via a generic registry.

**Key design principles:**
- Backend-only: the frontend never calls MCP remotes directly.
- Registry-driven: adding a new MCP server requires only a config entry + optional adapter.
- Backward compatible: MCP artifacts are delivered as optional `artifacts`/`tool_results` fields in the SSE `final` event. Old frontends simply ignore unknown fields.
- Disabled by default: `MCP_ENABLED=false` produces zero behavioral change.

### Configuration

**Global MCP settings** (in `.env`):
```
MCP_ENABLED=false
MCP_REMOTE_ENABLED=true
MCP_TIMEOUT_SECONDS=30
MCP_MAX_RETRIES=2
MCP_TOOL_INVOCATION_ENABLED=true
MCP_ARTIFACT_STORAGE_MODE=inline
MCP_TOTAL_BUDGET_SECONDS=60
MCP_TOOL_CACHE_TTL_SECONDS=300
MCP_MAX_ARTIFACT_CONTENT_BYTES=512000
```

**Server registry** (in `config/mcp_servers.json`):
```json
{
  "servers": [
    {
      "id": "excalidraw",
      "label": "Excalidraw Whiteboard",
      "url": "https://mcp.excalidraw.com",
      "enabled": true,
      "auth_strategy": null,
      "auth_secret_env": null,
      "timeout_seconds": 30,
      "tool_allowlist": [],
      "artifact_capabilities": ["excalidraw_json", "link", "svg", "png_url"],
      "health_check_interval_seconds": 300,
      "tags": ["drawing", "diagramming"]
    }
  ]
}
```

**Per-server env overrides** (optional):
```
MCP_SERVER_EXCALIDRAW_ENABLED=true
MCP_SERVER_EXCALIDRAW_URL=https://mcp.excalidraw.com
MCP_SERVER_EXCALIDRAW_TIMEOUT=30
MCP_SERVER_EXCALIDRAW_TOOL_ALLOWLIST=create-excalidraw,export-image
MCP_SERVER_EXCALIDRAW_AUTH_STRATEGY=bearer
MCP_SERVER_EXCALIDRAW_AUTH_SECRET_ENV=MCP_EXCALIDRAW_AUTH_TOKEN
```

### Architecture

The MCP module lives under `app/services/mcp/`:

```
app/services/mcp/
├── __init__.py            # Public API (get_mcp_client_manager, normalize_tool_result)
├── config.py              # McpServerConfig, McpGlobalConfig, load_mcp_config()
├── connector.py           # McpRemoteConnector — single server connection via StreamableHTTP
├── client_manager.py      # McpClientManager — connection pool, invoke_tool()
├── tool_registry.py       # McpToolRegistry — tool discovery, caching, allowlist
├── artifact.py            # Artifact model, sanitization, McpToolResult
├── adapters/
│   ├── __init__.py        # ADAPTER_REGISTRY, get_result_adapter()
│   ├── base.py            # BaseResultAdapter, GenericResultAdapter
│   └── excalidraw.py      # ExcalidrawResultAdapter
└── exceptions.py          # McpConnectionError, McpToolError, McpTimeoutError
```

### API Response Contract

When MCP tools are invoked, the SSE `final` event includes optional fields:

```json
{
  "success": true,
  "reply": "Here is the diagram I created for you...",
  "usage": { "input_tokens": N, "output_tokens": N },
  "request_id": "uuid",
  "sources": [...],
  "assistant_meta": { ... },
  "retrieval": { ... },
  "artifacts": [
    {
      "id": "art_abc123",
      "type": "excalidraw",
      "title": "Architecture Diagram",
      "url": "https://excalidraw.com/#json=...",
      "preview_url": null,
      "metadata": { "tool_server": "excalidraw", "tool_name": "create-excalidraw" }
    }
  ],
  "tool_results": [
    {
      "tool_server_id": "excalidraw",
      "tool_name": "create-excalidraw",
      "status": "success",
      "duration_ms": 1200,
      "artifact_ids": ["art_abc123"]
    }
  ]
}
```

- `artifacts` and `tool_results` are **optional** — omitted when no MCP tools were used.
- Only artifacts with `safe=True` (sanitized) are included.
- Old frontend clients safely ignore these unknown fields.

### Frontend Artifact Rendering

The frontend renders artifact cards in assistant messages via the `ArtifactRenderer` component tree:

```
ArtifactRenderer/
├── ArtifactRenderer.jsx              # Type router
├── ArtifactCard.jsx                  # Shared card wrapper
├── ImageArtifactRenderer.jsx         # type: "image", "svg"
├── ExcalidrawArtifactRenderer.jsx    # type: "excalidraw"
├── LinkArtifactRenderer.jsx          # type: "link"
└── GenericToolResultRenderer.jsx     # default fallback
```

**Security model (frontend):**
- No `dangerouslySetInnerHTML` anywhere (0 matches).
- SVG always rendered via `<img src=...>` (never inline HTML).
- URL validation: Excalidraw URLs (https + domain), preview URLs (https/http/data:image), generic links (https-only).
- JSON content downloaded as blob, never parsed as HTML.
- All event handlers are React-managed — no inline handlers from MCP data.
- JSON content size capped at 500KB on the backend.

### Adding a New MCP Server

1. **Add config entry** to `config/mcp_servers.json` (id, url, enabled, timeout, tool_allowlist).
2. **Optionally add adapter** only if the tool's output needs custom normalization (most tools can use `GenericResultAdapter`).
3. **No frontend rewrite** if output maps to existing artifact types (image, link, text, JSON).
4. **No backend pipeline changes** — the MCP client manager handles new tools automatically.

### Security Notes

- **No secret forwarding**: User JWT or internal secrets are never sent to MCP remotes.
- **Tool allowlist**: Only pre-approved tools can be invoked.
- **Output sanitization**: All MCP tool outputs are sanitized before inclusion in the response.
- **No raw HTML/SVG passthrough**: SVG content is sanitized (script/event handler stripping).
- **Rate limiting**: MCP calls count toward existing rate limits.
- **Audit logging**: Every MCP tool invocation is logged.
- **No user-controlled MCP URLs**: Server URLs are admin-controlled via config file and env vars.

### Known Limitations

- **LLM tool-use loop (Option A) not implemented**: The `artifacts`/`tool_results` response fields are structurally ready, but tool data must be manually constructed or injected until Option A is implemented. Option B (system-prompt-driven approach) is currently active.
- **3 pre-existing backend test collection failures**: FastAPI/Starlette `on_startup` version incompatibility — not caused by MCP work.
- **Real connectivity test to `https://mcp.excalidraw.com` not automated**: All MCP tests run without network (fully mocked).

- app DB có thể dùng `DATABASE_URL` tổng quát; nếu không set thì backend vẫn fallback về cấu hình MySQL cũ
- object storage hỗ trợ `OBJECT_STORAGE_PROVIDER=local` mặc định và có thể chuyển sang `s3`/`minio` để lưu file gốc + normalized text snapshot
- vector store hỗ trợ `VECTOR_STORE_PROVIDER=database` mặc định và có thể chuyển sang `qdrant` để retrieval top-k chạy qua vector DB thay vì quét chunks trong SQL
- luồng mặc định hiện đã được re-validate sau thay đổi kiến trúc bằng `scripts/knowledge_smoke_test.py` và `scripts/rag_chat_smoke_test.py`

### Cách chạy local với Postgres + MinIO + Qdrant

Repo hiện có sẵn stack local mẫu tại `deploy/docker-compose.local-rag.yml` và file env mẫu tại `.env.local-rag.example`.

Stack local này cũng dựng `rag-core` như một service riêng tại `http://127.0.0.1:8010`. Nếu backend chạy trực tiếp trên Windows và muốn dùng API mode, đặt:

```env
RAG_CORE_MODE=api
RAG_CORE_BASE_URL=http://127.0.0.1:8010
RAG_CORE_API_KEY=local-rag-core-dev-key
```

Nếu muốn rollback nhanh về đường in-process cũ, đặt `RAG_CORE_MODE=library`.

Quy ước môi trường cho backend:

- Chỉ dùng một Python env duy nhất của repo: `c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe`
- Không dùng `DominicProject` hoặc gọi trần `uvicorn`, `alembic`, `pip` từ PATH toàn cục vì dễ lệch dependency với repo
- Trong VS Code, backend nên được chạy qua task hoặc script `scripts/dev_backend.ps1` để luôn khóa đúng interpreter

1. Cài Docker Desktop và bảo đảm lệnh `docker compose` chạy được.
2. Từ thư mục repo backend, copy `.env.local-rag.example` thành `.env` rồi điền `ANTHROPIC_API_KEY` thật. Nếu muốn bật AI web search, điền thêm `TAVILY_API_KEY` và đổi `WEB_SEARCH_ENABLED=true`.
3. Dựng hạ tầng local:

```powershell
docker compose -f deploy/docker-compose.local-rag.yml up -d
```

4. Kiểm tra các service đã lên:

```powershell
docker compose -f deploy/docker-compose.local-rag.yml ps
```

5. Chạy migration backend để tạo schema trong Postgres:

```powershell
c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe -m alembic upgrade head
```

6. Chạy backend:

```powershell
c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

7. Mốc kiểm tra sau khi lên local stack:
  - Postgres: `127.0.0.1:5432`
  - Qdrant API: `http://127.0.0.1:6333/dashboard`
  - MinIO API: `http://127.0.0.1:9000`
  - MinIO Console: `http://127.0.0.1:9001`
  - rag-core API: `http://127.0.0.1:8010/health`
8. Tài khoản mặc định của MinIO local mẫu:
  - access key: `minioadmin`
  - secret key: `minioadmin123`
  - bucket: `dominic-knowledge`
9. Sau khi backend chạy, hãy upload lại ít nhất một tài liệu để tài liệu đó được ghi vào Postgres metadata, MinIO artifact store và Qdrant collection.
10. Nếu muốn dừng stack local:

```powershell
docker compose -f deploy/docker-compose.local-rag.yml down
```

Muốn xóa toàn bộ data local để làm sạch từ đầu:

```powershell
docker compose -f deploy/docker-compose.local-rag.yml down -v
```

### Local Redis + Celery worker cho async ingestion

Redis/Celery async ingestion is optional and guarded by `CELERY_ENABLED`. The frontend remains on the sync path unless a caller explicitly sends `async_index=true`.

Start the local RAG stack with Redis and the Docker Celery worker:

```powershell
docker compose -f deploy/docker-compose.local-rag.yml -f deploy/docker-compose.local-redis.yml up -d --build
```

Validate the merged compose config without starting containers:

```powershell
docker compose --env-file .env.local-redis.example -f deploy/docker-compose.local-rag.yml -f deploy/docker-compose.local-redis.yml config
```

Backend env vars for host-side local async testing:

```dotenv
CELERY_ENABLED=true
REDIS_PASSWORD=redis_dev_password
CELERY_BROKER_URL=redis://:redis_dev_password@127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://:redis_dev_password@127.0.0.1:6379/1
```

Docker worker command used by compose:

```powershell
celery -A app.worker.celery_app worker --loglevel=info --concurrency=2 -Q celery,ingestion
```

If you run a worker directly on Windows instead of Docker, use the solo pool:

```powershell
.\.venv\Scripts\celery -A app.worker.celery_app worker --loglevel=info --pool=solo -Q celery,ingestion
```

Available smoke scripts:

```powershell
# API async-ingestion smoke (requires backend/auth token/running stack for live use)
.\.venv\Scripts\python.exe scripts\celery_async_ingest_smoke_test.py --help

# Redis broker smoke (requires Redis running and CELERY_BROKER_URL set or --url supplied)
.\.venv\Scripts\python.exe scripts\smoke_redis.py --help

# Celery worker liveness smoke (requires at least one running worker for live PASS)
.\.venv\Scripts\python.exe scripts\smoke_celery_worker.py --help
```

For live local Redis/worker checks after the stack is up:

```powershell
$env:CELERY_BROKER_URL="redis://:redis_dev_password@127.0.0.1:6379/0"
.\.venv\Scripts\python.exe scripts\smoke_redis.py
.\.venv\Scripts\python.exe scripts\smoke_celery_worker.py
```

Stop the local Redis/Celery stack:

```powershell
docker compose -f deploy/docker-compose.local-rag.yml -f deploy/docker-compose.local-redis.yml down
```

Rollback switch: set `CELERY_ENABLED=false` and restart the backend. The default sync ingestion path remains available.

### Local Nginx Docker proxy cho frontend streaming

Local development can run through:

```text
Frontend Vite local -> Nginx Docker local -> FastAPI local
```

FE and BE stay outside Docker. Docker only runs the Nginx proxy so local development can exercise a production-like reverse proxy path while preserving SSE streaming.

1. Start the backend outside Docker on `127.0.0.1:8000`:

```powershell
.\scripts\dev_backend.ps1
```

2. Start the local Nginx proxy:

```powershell
docker compose -f deploy/docker-compose.local-nginx.yml up -d
```

3. Check Nginx and FastAPI through the proxy:

```powershell
curl http://127.0.0.1:8080/nginx-healthz
curl http://127.0.0.1:8080/health
```

4. Start the frontend outside Docker in `Dominic/chatbot-ui` with:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8080
VITE_API_TIMEOUT_MS=120000
```

To bypass Nginx and call the backend directly, use `VITE_API_BASE_URL=http://127.0.0.1:8000`.

The local Nginx config is `deploy/nginx/local-api-proxy.conf`; the dedicated compose file is `deploy/docker-compose.local-nginx.yml`. It proxies `/api/` and `/health` to `http://host.docker.internal:8000` and disables proxy buffering/cache for SSE streaming.

### Migrate dữ liệu từ MySQL cũ sang Postgres mới

Schema Postgres được tạo bởi Alembic, nhưng dữ liệu cũ từ MySQL phải được copy riêng bằng script one-off `scripts/migrate_mysql_to_postgres.py`.

1. Điền URL MySQL cũ vào biến `SOURCE_DATABASE_URL`.
2. Nếu muốn, điền `TARGET_DATABASE_URL`; nếu bỏ trống thì script sẽ dùng `DATABASE_URL` hiện tại của app.
3. Chạy lệnh migrate:

```powershell
c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/migrate_mysql_to_postgres.py --truncate-target
```

4. Script sẽ copy các bảng theo thứ tự khóa ngoại: `users`, `chat_sessions`, `chat_summaries`, `messages`, `knowledge_documents`, `knowledge_chunks`, `ingestion_jobs`, `retrieval_events`, `answer_citations`, `audit_logs`.
5. Sau khi copy xong, script sẽ tự reset sequence `id` trong Postgres để insert mới không bị đụng khóa chính cũ.

### Backfill knowledge cũ sang MinIO + Qdrant

Sau khi migrate relational data từ MySQL sang Postgres, các knowledge document cũ vẫn cần được backfill vào object storage và vector store để kiến trúc 3-storage chạy đầy đủ cho dữ liệu lịch sử.

1. Đảm bảo `.env` đang dùng:
  - `DATABASE_URL` trỏ Postgres mới
  - `OBJECT_STORAGE_PROVIDER=minio` hoặc `s3`
  - `VECTOR_STORE_PROVIDER=qdrant`
2. Chạy script backfill:

```powershell
c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/backfill_three_storage.py
```

3. Script sẽ:
  - ghi normalized text snapshot của document vào object storage
  - ghi `source-status/unavailable.json` cho document legacy không còn source bytes gốc
  - upsert lại vectors của chunks hiện có vào Qdrant mà không đổi `chunk_id`
4. Nếu chỉ muốn xử lý một document:

```powershell
c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/backfill_three_storage.py --document-id 8
```

5. Nếu chỉ muốn xử lý theo owner:

```powershell
c:/Users/Admin/Documents/GitHub/DominicBE/.venv/Scripts/python.exe scripts/backfill_three_storage.py --owner test_user
```

Ngoài script CLI, admin cũng có thể trigger cùng logic qua API:

```http
POST /api/knowledge/admin/backfill-three-storage
Authorization: Bearer <admin-token>
Content-Type: application/json

{
  "document_ids": [8],
  "owner_username": null,
  "limit": 10,
  "write_object_artifacts": true,
  "upsert_vectors": true,
  "write_source_manifest": true,
  "fail_fast": false
}
```

Response trả về số document được chọn, số thành công/thất bại, tổng vector points đã upsert, và kết quả chi tiết theo từng document.

### RAG retrieval quality metadata, section retrieval, và re-index/backfill

Các thay đổi RAG retrieval quality hiện tại là additive và không yêu cầu schema migration. Retrieval/chat metadata có thể bao gồm:

- `rag_mode`: `direct_chat`, `document_rag`, `session_rag`, `section_rag`, hoặc `indexing_pending`
- `retrieval_scope`: `none`, `document`, `session`, hoặc `global`
- `selected_document_id`, `session_id`, `section_key`, `section_confidence`
- `vector_store_attempted`, `vector_store_failed`, `vector_store_error_type`, `fallback_reason`

Khi tài liệu mới được ingest hoặc tài liệu cũ được re-index bằng custom pipeline, chunk metadata có thể có thêm `section_key`, `section_title`, `section_level`, `section_order`, `page_number`, `page_range`, `char_start`, và `char_end`. Các trường này được dùng để trả lời câu hỏi dạng section-level, ví dụ `Bài thực hành số 4 có mấy bài, tóm tắt từng bài`, bằng `rag_mode=section_rag` và evidence được sắp theo đúng thứ tự chunk trong section.

Lưu ý quan trọng cho dữ liệu cũ:

- Các indexed document cũ không tự động có `section_key`, `page_number`, `page_range`, `char_start`, hoặc `char_end`.
- Section/page/span metadata chỉ xuất hiện sau khi tài liệu được ingest mới hoặc re-index từ `raw_text` hiện có.
- Nếu document không còn `raw_text`, không đoán metadata từ các chunk cũ; hãy re-upload hoặc khôi phục source text rồi ingest/re-index.
- Session-bound chat không fallback sang global document khi session document chưa indexed; đây là safety fix có chủ ý. Metadata sẽ thể hiện `rag_mode=indexing_pending` và `fallback_reason=no_indexed_session_documents`.

Dry-run kiểm tra document cũ cần re-index để có section/span/page metadata:

```powershell
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe scripts\backfill_section_metadata.py --dry-run --limit 50
```

Chỉ kiểm tra một document:

```powershell
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe scripts\backfill_section_metadata.py --dry-run --document-id 8
```

Apply re-index theo batch sau khi dry-run đã được review:

```powershell
C:\Users\Admin\Documents\DominicChatbot\DominicBE\.venv\Scripts\python.exe scripts\backfill_section_metadata.py --apply --limit 50
```

Flow này gọi cùng re-index pipeline với endpoint `POST /api/v1/knowledge/documents/{doc_id}/reindex`, nên chunk content/vector sẽ được tạo lại từ `raw_text` và metadata mới sẽ được tính lại một cách idempotent. Dùng `--owner <username>` hoặc `--document-id <id>` để giữ batch nhỏ và không mở rộng phạm vi ngoài owner/document được chọn.

Smoke/golden coverage:

- `scripts/knowledge_smoke_test.py` ingest tài liệu synthetic mới có heading `Bài thực hành số 4`, xác nhận chunk metadata có `section_key=bai-thuc-hanh-so-4`, và xác nhận `/api/v1/knowledge/search` trả `rag_mode=section_rag` cho câu hỏi count/summary.
- `scripts/rag_chat_smoke_test.py` xác nhận grounded chat vẫn hoạt động và thêm kiểm tra chat section-level với cùng tài liệu synthetic.
- `scripts/data/rag_golden_set.json` có thêm case Vietnamese `Bài thực hành số 4 có mấy bài, tóm tắt từng bài` và English `Practice Lesson 4`; các case này yêu cầu fresh ingestion hoặc re-index/backfill trước khi chạy trên dữ liệu lâu đời.

Rollout khuyến nghị:

1. Deploy code và chạy regression/smoke trên môi trường staging với document ingest mới.
2. Chạy `backfill_section_metadata.py --dry-run` để ước lượng số document cần re-index và các document thiếu `raw_text`.
3. Re-index theo batch nhỏ bằng `--apply --limit <n>` hoặc endpoint reindex từng document; theo dõi lỗi và Qdrant/vector-store metrics.
4. Chạy lại smoke/golden cases section-level sau mỗi batch đại diện.
5. Monitor retrieval metadata: tỷ lệ `section_rag`, `indexing_pending`, `vector_store_failed`, và `fallback_reason`.

SQLite/PostgreSQL JSON lookup caveat: MVP section lookup vẫn giữ DB/session/access-control filter ở backend và lọc metadata portable. Nếu production traffic cho section retrieval cao, có thể cân nhắc index PostgreSQL expression cho `metadata_json.section_key` sau khi đã xác nhận query shape bằng `EXPLAIN`. Ví dụ PostgreSQL-only, không áp dụng cho SQLite và không chạy như migration không guarded:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_knowledge_chunks_metadata_section_key
ON knowledge_chunks ((metadata_json->>'section_key'));
```

Không cần xóa metadata khi rollback; có thể disable section retrieval branch hoặc bỏ qua các key JSON additive. Nếu cần rollback dữ liệu, re-index lại bằng pipeline cũ hoặc hard-delete/re-upload theo quy trình vận hành hiện có.

---

## Dockerized AWS deployment hiện tại

### Đã làm được

- backend đã có `Dockerfile` production và entrypoint tự chạy `alembic upgrade head`
- frontend đã có `Dockerfile` multi-stage để build Vite và phục vụ bằng Nginx
- repo backend có `deploy/docker-compose.ec2.yml` để dựng `frontend + backend + postgres + minio + qdrant + redis + celery-worker`
- Redis trong EC2 compose chạy nội bộ Docker (`expose: ["6379"]`, không có host `ports:`), yêu cầu `REDIS_PASSWORD`, dùng AOF (`--appendonly yes`) và lưu vào `redis_data:/data`
- celery-worker dùng cùng backend image, không expose port, và chạy `celery -A app.worker.celery_app worker --loglevel=info --concurrency=2 -Q celery,ingestion`
- repo backend có Nginx config mẫu và systemd service mẫu cho EC2

### Chưa bao gồm trong stack này

- `9router` chưa được đóng gói trong compose production này
- nếu muốn chat hoạt động, bạn phải cấu hình provider registry mặc định:
  `LLM_DEFAULT_PROVIDER=ninerouter`, `LLM_DEFAULT_MODEL`, `NINEROUTER_BASE_URL`, và `NINEROUTER_API_KEY`
- thêm provider/model OpenAI-compatible mới bằng `LLM_PROVIDER_CATALOG_JSON` và các env base URL/API key tương ứng; frontend lấy model picker từ `/api/v1/chat/models`
- mỗi model trong `LLM_PROVIDER_CATALOG_JSON` có thể khai báo `contextWindow` và `maxOutputTokens`; nếu bỏ trống, backend dùng fallback `LLM_CONTEXT_WINDOW` và `MAX_OUTPUT_TOKENS`

### Cần làm tiếp khi triển khai

- clone `DominicBE` và `Dominic` thành hai thư mục sibling trên EC2
- copy `.env.ec2.example` thành `.env.ec2` và điền secret/domain thật, gồm `REDIS_PASSWORD=<strong 32+ char secret>` và `CELERY_ENABLED=true` sau khi sẵn sàng bật async worker
- xác nhận AWS Security Group không có inbound TCP/6379; Phase F static docs không xác minh được AWS SG khi không có EC2/AWS access
- chạy `BACKEND_ENV_FILE=../.env.ec2.example docker compose --env-file .env.ec2.example -f deploy/docker-compose.ec2.yml config` để kiểm tra compose trước khi deploy
- chạy `docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml up -d --build`
- sau deploy, smoke check `docker compose ps`, `curl http://127.0.0.1:8000/health`, `docker exec dominic-redis redis-cli -a "$REDIS_PASSWORD" ping`, worker logs, và xác nhận host không listen public `:6379`
- rollback nhanh khi Redis/worker lỗi: đặt `CELERY_ENABLED=false` và restart backend; sync/default ingestion vẫn hoạt động
- cấu hình Nginx host cho `dominicapp.dev` và `api.dominicapp.dev`
- các lần update sau có thể dùng `./scripts/deploy_ec2.sh` trên EC2 thay cho việc gõ lại từng lệnh

Guide chi tiết từng bước nằm ở `DEPLOY_AWS_EC2_DOCKER.md`.

### Nginx/SSE streaming requirement

The streaming chat endpoint `POST /api/v1/chat/stream` uses Server-Sent Events. Nginx buffers proxied responses by default, which can make a correctly streaming backend appear broken because token chunks are grouped and flushed only after buffering or completion.

Every Nginx `location /` block that proxies Dominic traffic must preserve the existing `proxy_pass` upstream and include these SSE-safe directives:

```nginx
proxy_http_version 1.1;
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 300;
proxy_set_header Connection "";
```

Use the templates in `deploy/nginx/dominic.conf.example` or `deploy/nginx/dominic-docker-ec2.conf.example`, then validate and reload Nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

Verify streaming through the public proxy with `curl -N` and a valid bearer token:

```bash
curl -N -X POST https://api.dominicapp.dev/api/v1/chat/stream \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Stream a concise response."}'
```

Expected output order is `event: start`, then incremental `event: delta` chunks, then `event: final`. If CloudFront or another CDN is in front of Nginx, create a pass-through/no-cache behavior for `/api/v1/chat/stream` or otherwise exclude that path from CDN caching and response buffering.

---

# Legacy deployment guide (deprecated - MySQL + systemd)

Phần bên dưới là guide cũ cho thời điểm backend còn đi theo hướng `MySQL + systemd` và frontend chưa được đưa về AWS.

Nếu bạn triển khai trạng thái code hiện tại, hãy ưu tiên dùng guide mới ở `DEPLOY_AWS_EC2_DOCKER.md`.

This backend is a FastAPI app using:
- FastAPI + Gunicorn/Uvicorn
- MySQL
- OpenAI-compatible LLM provider registry (`9router` là provider mặc định)

This guide is written for deploying to **AWS EC2 in Singapore (`ap-southeast-1`)**.
It assumes:
- backend repo: `DominicBE`
- frontend repo: `Dominic`
- you want to deploy **backend + database first** on one EC2 Linux server
- frontend may stay on another host for now, or move later

---

## 1. Recommended architecture

### Option A - easiest for now
- EC2 instance runs:
  - FastAPI backend
  - MySQL database
  - Nginx reverse proxy
- Frontend stays elsewhere and calls EC2 backend over HTTPS

### Option B - cleaner later
- EC2 instance runs backend + MySQL
- frontend is deployed separately
- backend is exposed via domain like `https://api.yourdomain.com`

For your current phase, **Option A is the simplest**.

---

## 2. What changed in this project for EC2

The project has been adjusted so it is less Azure-specific and more suitable for EC2:

- `app/main.py`
  - removed Azure-specific assumptions
  - CORS now depends mainly on `CORS_ORIGINS`
  - `/debug/env` is disabled unless `ENABLE_DEBUG_ENV=true`
- `app/core/database.py`
  - supports `DB_SSL`, `DB_SSL_CA`, `DB_CHARSET`
  - supports configurable pool settings
  - builds DB URL safely even if password contains special characters
- `app/services/chat_service.py`
  - deployment messages are generic instead of Azure-only
  - supports `ANTHROPIC_FORCE_IPV4=true` for EC2 environments where IPv6 resolution exists but outbound IPv6 connectivity is broken
- `startup.sh`
  - now supports `HOST`, `PORT`, `WEB_CONCURRENCY`
- `.env.example`
  - updated for generic Linux/EC2 deployment

---

## 3. EC2 instance creation

Go to **AWS Console -> EC2 -> Instances -> Launch instances**.

Use these values:

### 3.1 Name
- `dominic-backend-sg`

### 3.2 AMI
- `Ubuntu Server 24.04 LTS` or `Ubuntu Server 22.04 LTS`

### 3.3 Instance type
- minimum: `t3.small`
- recommended if using MySQL + backend together: `t3.medium`

### 3.4 Key pair
- create or select an SSH key pair
- download the `.pem` file and keep it safe

### 3.5 Network settings
In the **Security group** section, allow:
- SSH: port `22` from **your own IP only**
- HTTP: port `80` from `0.0.0.0/0`
- HTTPS: port `443` from `0.0.0.0/0`

Do **not** open MySQL `3306` publicly if MySQL is on the same EC2.

### 3.6 Storage
- at least `20 GB`
- recommended `30 GB` if database is local

Then click **Launch instance**.

---

## 4. Optional but strongly recommended: Elastic IP

Go to:
- **AWS Console -> EC2 -> Elastic IPs**

Create an Elastic IP and attach it to your EC2 instance.

This gives you a stable public IP, so your frontend can call the backend reliably.

---

## 5. Connect to the server

From Windows PowerShell or Command Prompt:

```bash
ssh -i "C:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

If SSH fails because of key permissions on Windows, use PowerShell or Git Bash, or fix file permissions first.

---

## 6. Install system packages on Ubuntu

After logging into EC2, run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git nginx mysql-server pkg-config default-libmysqlclient-dev build-essential
```

Check versions:

```bash
python3 --version
nginx -v
mysql --version
```

---

## 7. Create application folder

On EC2:

```bash
mkdir -p /var/www
cd /var/www
sudo git clone https://github.com/admininistrator/DominicBE.git
sudo chown -R ubuntu:ubuntu /var/www/DominicBE
cd /var/www/DominicBE
```

If your repo is private, clone using SSH or a GitHub token.

---

## 8. Create Python virtual environment

Inside `/var/www/DominicBE`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Set up MySQL on the same EC2

### 9.1 Start MySQL

```bash
sudo systemctl enable mysql
sudo systemctl start mysql
sudo systemctl status mysql
```

### 9.2 Secure MySQL

```bash
sudo mysql_secure_installation
```

Recommended answers:
- validate password plugin: your choice
- remove anonymous users: `Y`
- disallow remote root login: `Y`
- remove test database: `Y`
- reload privilege tables: `Y`

### 9.3 Create database + app user

Open MySQL shell:

```bash
sudo mysql
```

Then run these SQL commands:

```sql
CREATE DATABASE chatbot_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'dominic'@'localhost' IDENTIFIED BY 'YOUR_STRONG_DB_PASSWORD';
GRANT ALL PRIVILEGES ON chatbot_db.* TO 'dominic'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

Important:
- because backend and MySQL are on the same EC2, use `'localhost'`
- do not expose MySQL publicly unless really necessary

---

## 10. Create backend environment file

On EC2:

```bash
cd /var/www/DominicBE
cp .env.example .env
nano .env
```

Paste/edit values like this:

```dotenv
APP_NAME=Dominic Backend
ENVIRONMENT=prod
DEBUG=false
ENABLE_DEBUG_ENV=false

AUTH_SECRET_KEY=replace_with_a_long_random_secret
AUTH_ALGORITHM=HS256
AUTH_ACCESS_TOKEN_EXPIRE_MINUTES=10080
AUTH_REFRESH_TOKEN_EXPIRE_MINUTES=43200
AUTH_PASSWORD_MIN_LENGTH=8
AUTH_PASSWORD_MAX_LENGTH=16

ANTHROPIC_API_KEY=your_real_anthropic_key
ANTHROPIC_MODEL=claude-3-5-haiku-latest
ANTHROPIC_BASE_URL=
ANTHROPIC_FORCE_IPV4=true

DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=dominic
DB_PASSWORD=YOUR_STRONG_DB_PASSWORD
DB_NAME=chatbot_db
DB_SSL=false
DB_SSL_CA=
DB_CHARSET=utf8mb4
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=300
DB_POOL_TIMEOUT=10

CORS_ORIGINS=https://your-frontend-domain.com,http://localhost:5173
ROLLING_WINDOW_HOURS=2
LLM_CONTEXT_WINDOW=200000
MAX_OUTPUT_TOKENS=5000
HOST=0.0.0.0
PORT=8000
WEB_CONCURRENCY=1
```

### Auth settings note

- `AUTH_SECRET_KEY` must be changed in every non-local environment
- the current auth flow issues both access and refresh tokens signed by `AUTH_SECRET_KEY`
- frontend currently stores the access token and refresh token in browser `localStorage`
- if you rotate `AUTH_SECRET_KEY`, all existing browser sessions will need to log in again

### What to enter in `CORS_ORIGINS`

If your frontend is still hosted elsewhere, enter the exact frontend origin, for example:

```dotenv
CORS_ORIGINS=https://black-desert-0b8b21b00.7.azurestaticapps.net
```

If you have both a production frontend and local dev frontend:

```dotenv
CORS_ORIGINS=https://black-desert-0b8b21b00.7.azurestaticapps.net,http://localhost:5173
```

Do not add path suffixes like `/api`.
Only origin, for example:
- correct: `https://example.com`
- wrong: `https://example.com/api/chat`

### If Anthropic fails on EC2 with connection errors

If these are true:

- `curl -4 -I https://api.anthropic.com` works
- `curl -6 -I https://api.anthropic.com` fails
- backend logs show `APIConnectionError` / `Connection error`

then keep this in `.env`:

```dotenv
ANTHROPIC_FORCE_IPV4=true
```

This project supports forcing the Anthropic SDK onto IPv4 to avoid broken IPv6 egress on some EC2 environments.

---

## 11. First backend run test

Inside `/var/www/DominicBE`:

```bash
cd /var/www/DominicBE
source .venv/bin/activate
chmod +x startup.sh
./startup.sh
```

If it starts correctly, open another SSH tab and test:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/postgres
curl http://127.0.0.1:8000/health/minio
curl http://127.0.0.1:8000/health/qdrant
```

Expected:

```json
{"service":"Dominic Backend","status":"running"}
```

and

```json
{"ok":true,"service":"Dominic Backend","dependencies":{"postgres":{"ok":true},"minio":{"ok":true},"qdrant":{"ok":true}}}
```

Use the three dedicated routes when you need to isolate whether a deployment issue is coming from Postgres, MinIO, or Qdrant specifically.

Press `Ctrl+C` to stop after confirming.

---

## 12. Create systemd service for backend

Create service file:

```bash
sudo nano /etc/systemd/system/dominic.service
```

Paste this:

```ini
[Unit]
Description=Dominic FastAPI backend
After=network.target mysql.service

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/var/www/DominicBE
EnvironmentFile=/var/www/DominicBE/.env
ExecStart=/var/www/DominicBE/.venv/bin/gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1 --timeout 120 --access-logfile - --error-logfile - --log-level info
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable dominic
sudo systemctl start dominic
sudo systemctl status dominic
```

To inspect logs:

```bash
sudo journalctl -u dominic -f
```

---

## 13. Configure Nginx reverse proxy

Create Nginx site:

```bash
sudo nano /etc/nginx/sites-available/dominic
```

Paste:

```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_EC2_IP;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }
}
```

Enable it:

```bash
sudo ln -s /etc/nginx/sites-available/dominic /etc/nginx/sites-enabled/dominic
sudo nginx -t
sudo systemctl restart nginx
```

Test publicly:

```bash
curl http://YOUR_DOMAIN_OR_EC2_IP/health
```

For streaming chat, Nginx must disable proxy buffering. Confirm the `location /` block includes `proxy_buffering off;`, `proxy_cache off;`, `proxy_read_timeout 300;`, and `proxy_set_header Connection "";`. Verify with `curl -N` after login/registering a user:

```bash
curl -N -X POST http://YOUR_DOMAIN_OR_EC2_IP/api/v1/chat/stream \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Stream a concise response."}'
```

If CloudFront or another CDN is used in front of this backend, exclude `/api/v1/chat/stream` from caching/buffering or configure SSE pass-through.

---

## 14. Add HTTPS with Let's Encrypt

If you have a domain name pointed to EC2, install SSL.

Install Certbot:

```bash
sudo apt install -y certbot python3-certbot-nginx
```

Issue certificate:

```bash
sudo certbot --nginx -d api.yourdomain.com
```

After success, your backend should be reachable at:

```text
https://api.yourdomain.com/health
```

If you do not have a domain yet, you can test with HTTP first using the EC2 public IP.

---

## 15. Frontend setting needed after backend is on EC2

If frontend stays on Azure Static Web Apps or another host, set:

```dotenv
VITE_API_BASE_URL=https://api.yourdomain.com
```

or if using raw IP temporarily:

```dotenv
VITE_API_BASE_URL=http://YOUR_EC2_PUBLIC_IP
```

If frontend and backend are later served from the same domain via Nginx, you can leave `VITE_API_BASE_URL` empty and let the browser call the same host.

---

## 16. If frontend is still on Azure Static Web Apps

Go to:
- **Azure Portal -> Static Web App -> Environment variables**

Set:

- Name: `VITE_API_BASE_URL`
- Value: `https://api.yourdomain.com`

Then redeploy frontend.

Also make sure backend `.env` has:

```dotenv
CORS_ORIGINS=https://black-desert-0b8b21b00.7.azurestaticapps.net
```

If you use a preview/staging frontend URL too, add both origins separated by commas.

---

## 17. How to seed a test user in MySQL

Your current backend expects users to exist in the `users` table.
Passwords should now be stored in `password_hash` using bcrypt.

Generate a bcrypt hash from the backend environment first:

```bash
cd /var/www/DominicBE
source .venv/bin/activate
python -c "from app.core.security import hash_password; print(hash_password('ChangeMe123!'))"
```

Copy the printed hash, then open MySQL:

Open MySQL:

```bash
mysql -u dominic -p
```

Then:

```sql
USE chatbot_db;
INSERT INTO users (username, password_hash, max_tokens_per_day)
VALUES ('test_user', '$2b$12$REPLACE_WITH_GENERATED_HASH', 10000);
```

If the user already exists:

```sql
UPDATE users
SET password_hash = '$2b$12$REPLACE_WITH_GENERATED_HASH',
    password = NULL
WHERE username = 'test_user';
```

Note:
- legacy rows that still have plaintext in `password` can log in once and will be auto-upgraded to `password_hash`
- do not manually paste plaintext passwords into `password_hash`
- avoid leading/trailing spaces in user passwords because the backend normalizes them before hashing and verification
- newly registered passwords are validated only by length: from 8 to 16 characters

### Alternative: create an account through the API

Instead of inserting directly into MySQL, you can create a user through the backend:

```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"phase1_user","password":"StrongPass1!","confirm_password":"StrongPass1!"}'
```

Successful responses return:
- `username`
- `access_token`
- `token_type=bearer`

---

## 18. Validation checklist after deployment

Run these checks in order.

### 18.1 Backend local on server

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/postgres
curl http://127.0.0.1:8000/health/minio
curl http://127.0.0.1:8000/health/qdrant
```

### 18.2 Backend through Nginx

```bash
curl http://YOUR_DOMAIN_OR_EC2_IP/health
curl http://YOUR_DOMAIN_OR_EC2_IP/health/postgres
curl http://YOUR_DOMAIN_OR_EC2_IP/health/minio
curl http://YOUR_DOMAIN_OR_EC2_IP/health/qdrant
```

### 18.3 Database connectivity

```bash
mysql -u dominic -p -e "USE chatbot_db; SHOW TABLES;"
```

### 18.4 Service logs

```bash
sudo journalctl -u dominic -n 100 --no-pager
```

### 18.5 Nginx logs

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 18.6 Browser test
- open frontend
- register a new account or login with an existing account
- create session
- send a prompt

If login works but sending prompt fails, inspect:
- `sudo journalctl -u dominic -f`
- Anthropic key/model
- outbound network from EC2

### 18.7 Auth smoke test

Run the built-in authentication smoke test:

```bash
cd /var/www/DominicBE
source .venv/bin/activate
python scripts/auth_smoke_test.py
```

Expected:

```text
AUTH_API_SMOKE_OK
```

### 18.8 Direct token check

After login or register, call `/api/v1/auth/me` using the returned bearer token:

```bash
curl http://127.0.0.1:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Expected:

```json
{"username":"phase1_user","role":"user"}
```

Refresh an expired access token with the refresh token:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"YOUR_REFRESH_TOKEN"}'
```

Logout now revokes both the current access token and any refresh token issued at the same auth version:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/logout \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 18.9 Knowledge ingestion + retrieval smoke test

Run the built-in knowledge pipeline smoke test:

```bash
cd /var/www/DominicBE
source .venv/bin/activate
python scripts/knowledge_smoke_test.py
```

Expected:

```text
KNOWLEDGE_API_SMOKE_OK
```

This validates the current Phase 2 MVP:

- ingest raw text into `knowledge_documents`
- upload supported files through `/api/knowledge/upload`
- chunking + local embedding metadata + `vector_id`
- searchable chunks through `/api/knowledge/search`
- document job history and reindex flow
- RAG retrieval quality metadata for newly ingested section documents, including `section_key`, `char_start`, `rag_mode=section_rag`, and `retrieval_scope=document` for the synthetic `Bài thực hành số 4` coverage

### 18.10 Direct Anthropic diagnostic on EC2

Run the built-in diagnostic script with the same `.env` used by systemd:

```bash
cd /var/www/DominicBE
source .venv/bin/activate
python scripts/test_anthropic_connection.py
```

This prints:

- whether the API key is loaded
- effective model/base URL
- whether `ANTHROPIC_FORCE_IPV4` is enabled
- `count_tokens` result
- `messages.create` result
- the full exception chain if the SDK still fails

---

## 19. Common problems

### Problem: frontend still calls `127.0.0.1:8000`
Cause:
- frontend build was created without correct `VITE_API_BASE_URL`

Fix:
- set `VITE_API_BASE_URL` in frontend environment
- rebuild/redeploy frontend

### Problem: CORS error
Cause:
- `CORS_ORIGINS` does not exactly match frontend origin

Fix:
- use exact origin only, such as:
  - `https://black-desert-0b8b21b00.7.azurestaticapps.net`
  - `http://localhost:5173`

### Problem: MySQL unknown database
Cause:
- `DB_NAME` does not exist

Fix:
- create the DB in MySQL
- verify `.env`

### Problem: backend starts but `/` returns `Not Found`
Cause:
- Nginx points to wrong upstream or app is not running

Fix:
- test `curl http://127.0.0.1:8000/`
- check `systemctl status dominic`
- check `nginx -t`

### Problem: Anthropic returns 403
Cause may be one of:
- model not enabled for the API key
- provider blocks region/egress IP
- billing/permissions issue
- wrong `ANTHROPIC_BASE_URL`

Fix:
- verify key on the EC2 server itself with a minimal Python test
- try another model that is definitely enabled
- verify outbound internet from the instance

---

## 20. How to update EC2 after you push new code to GitHub

If you already deployed the backend on EC2 by cloning the repo into:

```text
/var/www/DominicBE
```

then after every `git push`, update EC2 like this.

### 20.1 SSH into EC2

From Windows:

```bash
ssh -i "C:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

### 20.2 Go to project folder and pull latest code

```bash
cd /var/www/DominicBE
git status
git pull origin main
```

If your default branch is not `main`, replace it with the correct branch name.

### 20.3 Install new Python dependencies if `requirements.txt` changed

```bash
cd /var/www/DominicBE
source .venv/bin/activate
pip install -r requirements.txt
```

You can run this every time safely, even if dependencies did not change.

### 20.4 Restart backend service

```bash
sudo systemctl restart dominic
sudo systemctl status dominic
```

### 20.5 Check logs if needed

```bash
sudo journalctl -u dominic -n 100 --no-pager
sudo journalctl -u dominic -f
```

### 20.6 Verify backend is live

On the server:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/postgres
curl http://127.0.0.1:8000/health/minio
curl http://127.0.0.1:8000/health/qdrant
```

If using Nginx publicly:

```bash
curl http://YOUR_DOMAIN_OR_EC2_IP/health
curl http://YOUR_DOMAIN_OR_EC2_IP/health/postgres
curl http://YOUR_DOMAIN_OR_EC2_IP/health/minio
curl http://YOUR_DOMAIN_OR_EC2_IP/health/qdrant
```

If you changed only Python app code, normally you only need:

```bash
cd /var/www/DominicBE
git pull origin main
sudo systemctl restart dominic
```

### 20.7 When must you also restart Nginx?

Only restart Nginx if you changed Nginx config, domain, SSL, or reverse proxy settings:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

### 20.8 If `git pull` says you have local changes on EC2

Check what changed:

```bash
cd /var/www/DominicBE
git status
```

If the changed files are only local runtime files like `.env`, do not overwrite them.

If you accidentally edited tracked files on EC2 and want to discard them:

```bash
git reset --hard HEAD
git pull origin main
```

Warning: `git reset --hard` will delete uncommitted tracked changes.

### 20.9 If frontend also needs the new backend URL/config

If your frontend is still hosted on Azure Static Web Apps, remember:

- changing backend code on EC2 does **not** automatically rebuild frontend
- if frontend env vars changed, you must redeploy frontend too

For example, if `VITE_API_BASE_URL` changed, you must trigger a new frontend build/deploy.

### 20.10 Recommended simple update workflow

Use this order whenever you release a backend change:

1. push code to GitHub
2. SSH into EC2
3. run `git pull origin main`
4. run `source .venv/bin/activate`
5. run `pip install -r requirements.txt`
6. run `sudo systemctl restart dominic`
7. run `curl http://127.0.0.1:8000/health`
8. test from frontend

### 20.11 Optional: automate deployment from GitHub to EC2 later

After your manual deploy flow is stable, you can automate it with:

- **GitHub Actions + SSH**: easiest practical option
- **AWS CodeDeploy**: more formal, more setup

The easiest later approach is GitHub Actions that SSHs into EC2 and runs:

```bash
cd /var/www/DominicBE
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart dominic
```

## 21. Recommended next step after backend is stable

After backend + DB are working on EC2, do one of these:

1. keep frontend on Azure and only update `VITE_API_BASE_URL`
2. move frontend to S3 + CloudFront
3. move frontend to the same EC2 and let Nginx serve both frontend and backend under one domain

If you want, the next step I can do is:
- prepare the project for **EC2 + Nginx + same-domain frontend/backend**, or
- prepare the project for **EC2 backend + RDS MySQL** instead of local MySQL.

