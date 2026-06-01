# DominicBE — Streaming Audit Report

**Date**: 2026-05-28 15:26 UTC+7  
**Auditor**: Global System Auditor (reviewer mode)  
**Scope**: End-to-end streaming pipeline from user prompt → provider → frontend (SSE)  
**Outcome**: ✅ STREAMING LOGIC IS CORRECT — ⚠️ NGINX CONFIG BLOCKS IT

---

## 1. Executive Summary

The DominicBE backend **correctly implements true token-by-token streaming** from the LLM provider to the FastAPI response layer. The Python code is architecturally sound for streaming. **However, the production nginx configuration is missing the critical `proxy_buffering off;` directive**, which means nginx will buffer the entire LLM response before forwarding it to the frontend — completely defeating the purpose of streaming and producing the exact UX problem described (wait for full output → render all at once).

---

## 2. Streaming Pipeline: Full Trace

### 2.1 Frontend → API

| Step | File | Key Detail |
|------|------|------------|
| Request | (`POST /api/v1/chat/stream`) | [`ChatRequest`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\schemas\chat_schemas.py:95) with `session_id`, `message`, optional `model`, `images`, etc. |
| Auth | [`_assert_same_user()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:61) | Validates JWT token, normalizes username |
| Response wrapper | [`StreamingResponse`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:296) | `media_type="text/event-stream"` with correct headers |

### 2.2 API → Chat Service (Generator Chain)

```
stream_message()                              # chat.py:259
  └─ event_stream()  (inner generator)        # chat.py:267
       └─ handle_chat_stream()                # chat_service.py:1563
            ├─ yield {event: "start", ...}    # chat_service.py:1590
            ├─ for chunk in stream_complete() # chat_service.py:1593
            │    └─ yield {event: "delta", ...}  # chat_service.py:1598-1600
            └─ yield {event: "final", ...}    # chat_service.py:1615
```

**Key observation**: The `for chunk in llm_provider.stream_complete()` loop at [`chat_service.py:1593`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1593) yields each delta **immediately** to the outer generator. This is genuine streaming — no accumulator, no wait-for-completion.

### 2.3 Chat Service → LLM Provider

[`stream_complete()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:838) has **three streaming paths**, all correct:

| Path | Trigger | Streaming Method | Verdict |
|------|---------|-----------------|--------|
| LiteLLM | Default (ninerouter/9Router) | `litellm.completion(stream=True)` → iterate chunks → `yield {"type": "delta", ...}` per chunk | ✅ |
| Direct HTTP Chat | `minimax-m2.7`, Kimi/Qwen with reasoning | `requests.post(stream=True)` → `iter_lines()` → SSE parse → `yield` per chunk | ✅ |
| Direct HTTP Responses | `gemini-*` models | `requests.post(stream=True)` → `iter_lines()` → SSE parse → `yield` per chunk | ✅ |

Each path:
- Uses `requests.post(..., stream=True)` or `litellm.completion(stream=True)` — not buffered
- Iterates line-by-line via [`_iter_openai_compatible_sse_payloads()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:651)
- Extracts `delta.content` from each SSE payload via [`_extract_text_delta_from_stream_payload()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:582)
- Yields each delta individually — per-token granularity

### 2.4 API → HTTP Response

[`_sse_event()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:38):
```python
def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
```
Produces standard SSE format:
```
event: delta
data: {"text": "Chào", "request_id": "uuid-here"}

event: delta
data: {"text": " bạn", "request_id": "uuid-here"}

...

event: final
data: {"success": true, "reply": "Chào bạn...", ...}
```

### 2.5 HTTP Headers on StreamingResponse

From [`chat.py:299-303`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:299):

| Header | Value | Purpose |
|--------|-------|---------|
| `Cache-Control` | `no-cache` | Prevents client/proxy caching |
| `Connection` | `keep-alive` | Persistent connection for stream |
| `X-Accel-Buffering` | `no` | Instructs nginx not to buffer (when nginx is configured to honor it) |

---

## 3. Issues Found

### 🔴 CRITICAL: Nginx `proxy_buffering` is OFF by default behavior, but nginx still buffers unless explicitly told not to

**File**: [`dominic.conf.example`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\deploy\nginx\dominic.conf.example)  
**File**: [`dominic-docker-ec2.conf.example`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\deploy\nginx\dominic-docker-ec2.conf.example)

**Problem**: Neither nginx config declares `proxy_buffering off;` in the API server block. Nginx's default is `proxy_buffering on;`. This causes nginx to:

1. Accumulate the entire SSE stream from the backend in memory
2. Only forward to the client (frontend) when the backend connection closes or the buffer fills
3. Result: **Frontend receives the full response all at once** — exactly the UX problem described

**Evidence**: 
- `dominic.conf.example` (lines 7-14): No `proxy_buffering` directive
- `dominic-docker-ec2.conf.example` (lines 25-33): No `proxy_buffering` directive for the API server block

**Fix** (required in both configs):
```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_buffering off;                    # ← ADD THIS
    proxy_read_timeout 300;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### 🟡 MEDIUM: The `Connection` header in the API streaming endpoint may conflict

At [`chat.py:301`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:301), the `Connection: keep-alive` header is set. When nginx has `proxy_buffering off;` but does not have `proxy_set_header Connection "";`, nginx may pass through conflicting connection headers. This is a minor concern but worth noting.

### 🟡 MEDIUM: No streaming-specific tests

**File**: [`rag_chat_smoke_test.py`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\scripts\rag_chat_smoke_test.py)

The smoke test only tests the **non-streaming** [`POST /api/v1/chat/`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:218) endpoint. There is no test that:

- Calls `POST /api/v1/chat/stream`
- Reads the SSE response incrementally
- Verifies `event: start` arrives before any `event: delta`
- Verifies `event: delta` chunks contain incremental text (not the full response)
- Verifies `event: final` arrives with complete metadata
- Verifies the `request_id` is consistent across events

### 🔵 LOW: `start` event carries minimal metadata

At [`chat_service.py:1590`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1590), the start event only contains `request_id`. Since retrieval/web-search is already completed in [`_prepare_chat_turn()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1156) before streaming begins, the frontend could benefit from early metadata like `sources`, `retrieval` status, etc. This is a UX optimization opportunity, not a correctness issue.

### 🔵 LOW: DB session held open during entire stream

[`handle_chat_stream()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1563) receives `db: Session` from the dependency injection and holds it open across the entire streaming duration. This is acceptable for moderate usage but could be improved by splitting into two sessions: one for `_prepare_chat_turn()` and another for `_finalize_chat_turn()`.

---

## 4. Non-Streaming Fallback (Correctness Check)

The [`POST /api/v1/chat/`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:218) endpoint correctly calls [`handle_chat()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1499) which uses the non-streaming [`llm_provider.complete()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:400). This is a **separate endpoint** for non-streaming use cases. The streaming endpoint never falls back to non-streaming — it exclusively uses [`stream_complete()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:838).

This isolation is **correct architecture** — no accidental buffering from fallback logic.

---

## 5. Verdict Table

| Layer | Component | Streaming? | Notes |
|-------|-----------|------------|-------|
| Provider (LiteLLM) | [`stream_complete()` LiteLLM path](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:948) | ✅ | Yields per-chunk |
| Provider (Direct HTTP) | [`_stream_via_chat_http()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:759) | ✅ | `stream=True` + `iter_lines()` |
| Provider (Responses HTTP) | [`_stream_via_responses_http()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\llm_provider.py:669) | ✅ | `stream=True` + `iter_lines()` |
| Chat Service | [`handle_chat_stream()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1563) | ✅ | `yield` per delta, no aggregation |
| API Endpoint | [`stream_message()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\api\endpoints\chat.py:259) | ✅ | `StreamingResponse` wrapping generator |
| FastAPI Middleware | [`observability_middleware`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\main.py:405) | ✅ | `await call_next()` — doesn't buffer StreamingResponse |
| Dockerfile | `PYTHONUNBUFFERED=1` | ✅ | Prevents Python stdout buffering |
| Nginx (EC2/deploy) | `dominic*.conf.example` | ❌ | **Missing `proxy_buffering off;`** |

---

## 6. Action Items (Priority Order)

| # | Priority | Action | File(s) |
|---|----------|--------|---------|
| 1 | 🔴 CRITICAL | Add `proxy_buffering off;` to nginx API location block | [`dominic.conf.example`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\deploy\nginx\dominic.conf.example), [`dominic-docker-ec2.conf.example`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\deploy\nginx\dominic-docker-ec2.conf.example) |
| 2 | 🔴 CRITICAL | If using CloudFront/CDN in front of nginx, ensure it also passes through SSE without buffering | AWS Console / CDN config |
| 3 | 🟡 MEDIUM | Add `proxy_set_header Connection "";` in nginx config when using `proxy_buffering off;` | Same nginx configs |
| 4 | 🟡 MEDIUM | Add a streaming smoke test to verify SSE event order and incremental delivery | New test in `tests/` |
| 5 | 🔵 LOW | Consider yielding more metadata in the `start` event (sources, retrieval status) | [`handle_chat_stream()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1590) |
| 6 | 🔵 LOW | Split DB session into pre-stream and post-stream phases | [`handle_chat_stream()`](C:\Users\Admin\Documents\DominicChatbot\DominicBE\app\services\chat_service.py:1563) |

---

## 7. Conclusion

The **application-layer streaming logic is fully correct** and well-architected. Every layer from the LLM provider through to the FastAPI HTTP response uses true generator-based streaming with per-token granularity. The SSE protocol implementation is standards-compliant.

**The single blocking issue is the nginx reverse proxy configuration**, which defaults to `proxy_buffering on;` and will buffer the entire stream. Fixing that one directive will unlock the full streaming UX benefit that the backend code is already engineered to deliver.

---

*Report generated by Global System Auditor (reviewer mode) — 2026-05-28T08:26:00+07:00*
