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

- auth với register, login, bearer token, `/api/auth/me`
- password hashing, đổi mật khẩu, admin-issued reset token, public reset-password flow
- quản lý chat session, thống kê usage theo rolling window, conversation summary memory
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

- app DB có thể dùng `DATABASE_URL` tổng quát; nếu không set thì backend vẫn fallback về cấu hình MySQL cũ
- object storage hỗ trợ `OBJECT_STORAGE_PROVIDER=local` mặc định và có thể chuyển sang `s3`/`minio` để lưu file gốc + normalized text snapshot
- vector store hỗ trợ `VECTOR_STORE_PROVIDER=database` mặc định và có thể chuyển sang `qdrant` để retrieval top-k chạy qua vector DB thay vì quét chunks trong SQL
- luồng mặc định hiện đã được re-validate sau thay đổi kiến trúc bằng `scripts/knowledge_smoke_test.py` và `scripts/rag_chat_smoke_test.py`

### Cách chạy local với Postgres + MinIO + Qdrant

Repo hiện có sẵn stack local mẫu tại `deploy/docker-compose.local-rag.yml` và file env mẫu tại `.env.local-rag.example`.

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

---

## Dockerized AWS deployment hiện tại

### Đã làm được

- backend đã có `Dockerfile` production và entrypoint tự chạy `alembic upgrade head`
- frontend đã có `Dockerfile` multi-stage để build Vite và phục vụ bằng Nginx
- repo backend có `deploy/docker-compose.ec2.yml` để dựng `frontend + backend + postgres + minio + qdrant`
- repo backend có Nginx config mẫu và systemd service mẫu cho EC2

### Chưa bao gồm trong stack này

- `9router` chưa được đóng gói trong compose production này
- nếu muốn chat hoạt động, bạn vẫn phải cấu hình `GITHUB_COPILOT_API_KEY` và `NINEROUTER_BASE_URL` thật

### Cần làm tiếp khi triển khai

- clone `DominicBE` và `Dominic` thành hai thư mục sibling trên EC2
- copy `.env.ec2.example` thành `.env.ec2` và điền secret/domain thật
- chạy `docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml up -d --build`
- cấu hình Nginx host cho `dominicapp.dev` và `api.dominicapp.dev`
- các lần update sau có thể dùng `./scripts/deploy_ec2.sh` trên EC2 thay cho việc gõ lại từng lệnh

Guide chi tiết từng bước nằm ở `DEPLOY_AWS_EC2_DOCKER.md`.

---

# Legacy deployment guide (deprecated - MySQL + systemd)

Phần bên dưới là guide cũ cho thời điểm backend còn đi theo hướng `MySQL + systemd` và frontend chưa được đưa về AWS.

Nếu bạn triển khai trạng thái code hiện tại, hãy ưu tiên dùng guide mới ở `DEPLOY_AWS_EC2_DOCKER.md`.

This backend is a FastAPI app using:
- FastAPI + Gunicorn/Uvicorn
- MySQL
- LiteLLM provider layer (Anthropic mặc định)

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
AUTH_PASSWORD_MIN_LENGTH=8
AUTH_PASSWORD_MAX_LENGTH=16

ANTHROPIC_API_KEY=your_real_anthropic_key
ANTHROPIC_MODEL=claude-3-5-haiku-latest
ANTHROPIC_BASE_URL=
ANTHROPIC_FORCE_IPV4=true

DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=dominic
DB_PASSWORD=YOUR_STRONG_DB_PASSWORD
DB_NAME=chatbot_db
DB_SSL=false
DB_SSL_CA=
DB_CHARSET=utf8mb4
DB_POOL_RECYCLE=300
DB_POOL_TIMEOUT=10

CORS_ORIGINS=https://your-frontend-domain.com,http://localhost:5173
ROLLING_WINDOW_HOURS=2
MAX_OUTPUT_TOKENS=5000
HOST=0.0.0.0
PORT=8000
WEB_CONCURRENCY=1
```

### Auth settings note

- `AUTH_SECRET_KEY` must be changed in every non-local environment
- the current Phase 1 login flow uses bearer tokens signed by `AUTH_SECRET_KEY`
- frontend currently stores the access token in browser `localStorage`
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
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
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

After login or register, call `/api/auth/me` using the returned bearer token:

```bash
curl http://127.0.0.1:8000/api/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Expected:

```json
{"username":"phase1_user","role":"user"}
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

