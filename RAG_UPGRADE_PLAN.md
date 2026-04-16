# DominicBE -> RAG Upgrade Plan

## 1) Executive summary

Dự án hiện tại **chưa phải là RAG**. Đây là một hệ thống **chatbot có short-term memory + rolling summary** gọi Anthropic trực tiếp.

Hiện trạng phù hợp với giai đoạn MVP chat, nhưng để nâng cấp thành **chuẩn RAG production**, dự án cần bổ sung các năng lực cốt lõi sau:

1. **Knowledge ingestion**: nạp tài liệu vào hệ thống.
2. **Chunking + embedding**: băm/chia đoạn và tạo vector.
3. **Retrieval**: tìm đoạn liên quan theo truy vấn.
4. **Grounded generation**: sinh câu trả lời dựa trên context truy xuất được.
5. **Citations / sources**: trả nguồn để người dùng kiểm chứng.
6. **Evaluation + observability**: đo độ chính xác, độ phủ, hallucination, latency.
7. **Security + data governance**: auth, phân quyền tài liệu, vòng đời dữ liệu.

---

## 2) Current-state assessment

### Backend hiện có

- `app/services/chat_service.py`
  - Có `handle_chat(...)`
  - Có `_build_hybrid_context(...)`
  - Logic context hiện tại = **summary hội thoại + recent messages**
  - Chưa có retrieval theo knowledge base
- `app/models/chat_models.py`
  - Chỉ có user / session / message / chat summary
  - Chưa có bảng document / chunk / embedding index / citation
- `app/crud/crud_chat.py`
  - CRUD hiện thiên về lịch sử chat, token quota, auth đơn giản
  - Chưa có CRUD cho ingestion / indexing / retrieval
- `app/api/endpoints/chat.py`
  - API hiện chỉ phục vụ login, usage, session, chat
  - Chưa có API upload tài liệu, quản lý knowledge base, search, reindex
- `app/core/database.py`
  - Đang dùng MySQL + SQLAlchemy
  - Phù hợp lưu metadata, nhưng **không đủ tốt làm vector retrieval production** nếu không có thêm engine/vector extension phù hợp
- `app/main.py`
  - App FastAPI còn ở mức khá gọn, chưa có tracing/monitoring/evaluation hooks

### Frontend hiện có

- `chatbot-ui/src/App.jsx`
  - Tập trung vào login, session chat, render messages
- `chatbot-ui/src/components/ChatWindow/ChatWindow.jsx`
  - Chỉ render message list
- `chatbot-ui/src/components/MessageBubble/MessageBubble.jsx`
  - Có markdown và token usage
  - Chưa hiển thị citations / sources / retrieval metadata
- `chatbot-ui/src/service/chatApi.js`
  - Chỉ có chat/session/usage/login
  - Chưa có upload/search/source endpoints

### Kết luận hiện trạng

Hệ thống hiện tại là:

- **LLM chat app có memory summary**
- **Chưa có knowledge layer**
- **Chưa có retrieval pipeline**
- **Chưa có evidence-based answer flow**

=> Vì vậy, dự án đang ở mức **pre-RAG**.

---

## 3) Gaps so với một hệ thống RAG chuẩn

### A. Thiếu tầng dữ liệu tri thức
Cần có:
- `documents`
- `document_chunks`
- `retrieval_runs`
- `answer_citations`
- có thể thêm `knowledge_bases`, `ingestion_jobs`

### B. Thiếu ingestion pipeline
Cần có pipeline:
- upload file / import URL / import text
- parse file (`pdf`, `docx`, `txt`, `md`, `html`, `csv` tùy scope)
- clean text
- chunking
- embedding
- index

### C. Thiếu retrieval engine
Cần có:
- semantic retrieval
- top-k selection
- optional hybrid search (BM25 + vector)
- optional reranking
- threshold để tránh nhồi context rác

### D. Thiếu grounded generation
Hiện LLM trả lời dựa vào chat history.
RAG chuẩn cần:
- prompt có ngữ cảnh retrieved chunks
- instruction bắt buộc “only answer from retrieved evidence when relevant”
- fallback nếu không đủ bằng chứng

### E. Thiếu UX cho RAG
Cần có:
- upload tài liệu
- trạng thái indexing
- hiển thị nguồn trích dẫn
- xem đoạn nào được dùng để trả lời
- có thể thêm “knowledge base selector”

### F. Thiếu evaluation / observability
Cần có:
- log retrieval hit rate
- chunk relevance score
- answer groundedness
- latency từng stage: ingest / embed / retrieve / generate
- golden set để test chất lượng RAG

### G. Thiếu hardening production
Phát hiện trực tiếp từ code:
- `verify_user_credentials(...)` đang so sánh **plain text password**
- backend chưa có migration framework rõ ràng (hiện dựa vào `Base.metadata.create_all(...)`)
- README mô tả một số capability triển khai nhưng code hiện tại **chưa khớp hoàn toàn**:
  - `ENABLE_DEBUG_ENV` chưa thấy gating thực tế trong `app/main.py`
  - `DB_SSL_CA`, `DB_CHARSET`, `DB_POOL_RECYCLE`, `DB_POOL_TIMEOUT` chưa thấy support đầy đủ trong `app/core/database.py`
  - `HOST`, `WEB_CONCURRENCY` chưa được `startup.sh` dùng đầy đủ

---

## 4) Mức trưởng thành RAG đề xuất

### Level 0 - Current
- Chat + session
- rolling token quota
- conversation summary
- direct LLM call

### Level 1 - Basic RAG
- upload tài liệu
- parse -> chunk -> embed
- vector search top-k
- chat trả lời có nguồn

### Level 2 - Better RAG
- metadata filtering
- hybrid retrieval
- reranking
- citations theo chunk
- admin reindex

### Level 3 - Production RAG
- background jobs
- observability
- eval suite
- document ACL / multi-tenant
- prompt safety / guardrails
- scalable vector store

---

## 5) Kiến trúc RAG mục tiêu cho dự án này

## 5.1 Backend đề xuất

### API layer
Giữ FastAPI, mở rộng thêm:
- `POST /api/knowledge/documents/upload`
- `GET /api/knowledge/documents`
- `POST /api/knowledge/documents/{id}/reindex`
- `DELETE /api/knowledge/documents/{id}`
- `GET /api/knowledge/documents/{id}/chunks`

### Data model layer
Dùng MySQL tiếp tục cho metadata:
- users
- chat_sessions
- messages
- chat_summaries
- documents
- document_chunks
- retrieval_events
- answer_citations
- ingestion_jobs

### Vector store layer
Có 3 lựa chọn:

#### Option A - Nhanh nhất để đi tiếp
- MySQL giữ metadata
- vector store dùng **Qdrant**
- phù hợp khi muốn RAG đúng chuẩn nhanh mà ít đổi DB chính

#### Option B - Tạm thời cho local/dev
- vector lưu file local / FAISS
- không lý tưởng cho production nhiều người dùng

#### Option C - Chuẩn enterprise hơn về sau
- chuyển metadata + app DB sang PostgreSQL + `pgvector`
- phù hợp nếu muốn gom hệ thống về một DB mạnh cho search

**Khuyến nghị cho dự án hiện tại:**
- Giai đoạn 1: **MySQL + Qdrant**
- Không nên cố ép MySQL hiện tại thành vector DB chính cho production RAG.

### LLM / embedding layer
- giữ Anthropic cho generation
- thêm embedding provider riêng
- nên tách config:
  - `EMBEDDING_PROVIDER`
  - `EMBEDDING_MODEL`
  - `VECTOR_STORE_PROVIDER`
  - `VECTOR_STORE_URL`

---

## 6) Migration strategy: nâng cấp dần, không big-bang rewrite

## Phase 1 - Foundation hardening
Mục tiêu: ổn định nền trước khi thêm RAG.

### Việc cần làm
1. Thêm migration framework (Alembic).
2. Hash password bằng `passlib` / `bcrypt`.
3. Khóa hoặc cấu hình hóa debug endpoint.
4. Chuẩn hóa config theo env.
5. Tách service chat và service retrieval sau này cho dễ mở rộng.
6. Thêm cấu trúc model/schema sẵn cho knowledge.

### Kết quả phase 1
- backend an toàn hơn
- schema quản lý bài bản hơn
- sẵn nền để thêm ingestion

---

## Phase 2 - Knowledge ingestion MVP
Mục tiêu: hệ thống bắt đầu “đọc tài liệu”.

### Việc cần làm
1. Tạo models:
   - `Document`
   - `DocumentChunk`
   - `IngestionJob`
2. Tạo API upload text/file đơn giản.
3. Lưu file hoặc raw text.
4. Parse text.
5. Chunking theo rule cố định:
   - chunk size ~500-1000 tokens
   - overlap ~50-150 tokens
6. Tạo embedding cho từng chunk.
7. Đẩy chunk vectors vào vector store.

### Output kỳ vọng
- upload được 1 tài liệu
- xem được trạng thái indexed
- chunk đã searchable

---

## Phase 3 - Retrieval integration vào chat
Mục tiêu: chat bắt đầu dùng tài liệu khi trả lời.

### Việc cần làm
1. Tạo `retrieval_service.py`.
2. Truy vấn embedding từ câu hỏi user.
3. Search top-k chunks.
4. Optional metadata filter theo user / knowledge base.
5. Build context block từ chunks tìm được.
6. Sửa `handle_chat(...)` để context gồm:
   - conversation summary
   - recent messages
   - retrieved chunks
7. Trả về metadata của sources trong response.

### Output kỳ vọng
- chat response có:
  - `reply`
  - `usage`
  - `sources[]`
  - `request_id`

---

## Phase 4 - RAG UX trên frontend
Mục tiêu: người dùng thấy rõ chatbot đang trả lời dựa trên nguồn nào.

### Việc cần làm
1. Thêm trang / panel quản lý tài liệu.
2. Upload file/text.
3. Hiển thị trạng thái indexing.
4. Thêm render citations trong `MessageBubble`.
5. Có thể mở rộng `ChatWindow` để hiện:
   - top sources
   - score
   - chunk excerpt

### Output kỳ vọng
- user upload được knowledge
- user thấy nguồn trích dẫn trong câu trả lời

---

## Phase 5 - RAG quality improvements
Mục tiêu: từ “có RAG” sang “RAG dùng được tốt”.

### Việc cần làm
1. Query rewriting.
2. Hybrid retrieval.
3. Reranking.
4. Deduplicate chunks.
5. Context packing tối ưu token budget.
6. Fallback nếu retrieval yếu.
7. Guardrails:
   - nếu không có bằng chứng, nói rõ không chắc chắn
   - tránh bịa nguồn

---

## Phase 6 - Production readiness
Mục tiêu: hệ thống đủ chuẩn vận hành thực tế.

### Việc cần làm
1. Background worker cho indexing.
2. Retry policy cho embedding/indexing.
3. Metrics + tracing.
4. Evaluation dataset.
5. Multi-tenant document access.
6. Soft delete / versioning tài liệu.
7. Audit logs.
8. Cost dashboard.

---

## 7) Thiết kế schema tối thiểu cho RAG MVP

### documents
- id
- owner_username
- title
- source_type (`upload`, `text`, `url`)
- source_uri
- mime_type
- status (`uploaded`, `processing`, `indexed`, `failed`)
- checksum
- created_at
- updated_at

### document_chunks
- id
- document_id
- chunk_index
- content
- token_count
- embedding_model
- vector_id
- metadata_json
- created_at

### retrieval_events
- id
- username
- session_id
- query_text
- top_k
- latency_ms
- created_at

### answer_citations
- id
- request_id
- document_id
- chunk_id
- rank
- score
- quoted_text
- created_at

---

## 8) Thay đổi contract API đề xuất

### Chat response mới
```json
{
  "success": true,
  "reply": "...",
  "usage": {
    "input_tokens": 123,
    "output_tokens": 456
  },
  "request_id": "uuid",
  "sources": [
    {
      "document_id": 1,
      "chunk_id": 10,
      "title": "Policy.pdf",
      "score": 0.91,
      "snippet": "..."
    }
  ],
  "retrieval": {
    "used": true,
    "top_k": 5,
    "returned": 3
  }
}
```

### Upload response mới
```json
{
  "id": 1,
  "title": "Product FAQ",
  "status": "processing"
}
```

---

## 9) Ưu tiên kỹ thuật nên làm ngay

### Ưu tiên P0
1. Chuẩn hóa migration.
2. Sửa auth plain text password.
3. Khóa debug endpoint.
4. Đồng bộ README với code thực tế.

### Ưu tiên P1
1. Thêm models/schema cho documents/chunks.
2. Thêm config cho embedding/vector store.
3. Tạo ingestion MVP bằng text upload trước.

### Ưu tiên P2
1. Tích hợp retrieval vào `handle_chat(...)`.
2. Trả citations trong API.
3. Render sources ở frontend.

---

## 10) Khuyến nghị triển khai thực tế cho repo này

### Nên giữ
- FastAPI
- React/Vite frontend
- MySQL cho user/chat metadata
- Anthropic cho generation

### Nên bổ sung
- Alembic
- password hashing
- Qdrant hoặc pgvector
- ingestion service
- retrieval service
- citation models
- evaluation scripts

### Không nên làm ngay
- full agentic workflow quá sớm
- multi-model routing quá sớm
- OCR/document intelligence quá sớm
- phức tạp hóa retrieval trước khi có baseline

---

## 11) Definition of done cho “RAG MVP đầu tiên”

Dự án được xem là đạt **RAG MVP** khi:

1. Có thể upload ít nhất 1 nguồn tri thức.
2. Hệ thống chunk + embed + index thành công.
3. Câu hỏi user có retrieval top-k.
4. Prompt generation có context từ chunks.
5. Response trả về nguồn.
6. Frontend hiển thị citations.
7. Nếu không có bằng chứng phù hợp, model không trả lời như chắc chắn.

---

## 12) Đề xuất phase tiếp theo để bắt đầu code

### Phase kế tiếp nên làm ngay
**Phase 1: Foundation hardening + RAG skeleton**

Cụ thể:
1. thêm config cho RAG
2. thêm models `Document`, `DocumentChunk`
3. thêm schema response có `sources`
4. thêm khung `retrieval_service.py`
5. chưa cần vector store thật ở bước đầu
6. giữ backward-compatible với chat hiện tại

Đây là điểm bắt đầu đúng vì:
- ít phá hệ thống đang chạy
- tạo được bộ khung cho RAG
- sau đó mới nối ingestion và retrieval thật

---

## 13) Đề xuất sprint chia nhỏ

### Sprint 1
- hardening nền
- thêm schema/model RAG skeleton

### Sprint 2
- upload text document
- chunking + lưu DB

### Sprint 3
- embedding + vector store
- retrieval top-k

### Sprint 4
- chat grounded + citations
- frontend sources UI

### Sprint 5
- evaluation + monitoring
- tuning quality

---

## 14) Tổng kết

Hiện tại dự án **chưa là RAG**, nhưng kiến trúc hiện có đủ nhẹ để nâng cấp dần theo hướng đúng.

Hướng đi phù hợp nhất là:

1. **Hardening nền backend trước**
2. **Thêm knowledge ingestion skeleton**
3. **Nối retrieval vào chat**
4. **Hiển thị citations ở frontend**
5. **Tối ưu chất lượng và vận hành**

Nếu tiếp tục theo roadmap này, dự án có thể đi từ:
- chatbot dùng chat history
->
- chatbot có knowledge grounding
->
- production-grade RAG assistant

