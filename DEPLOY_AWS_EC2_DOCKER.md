# Dominic Deployment Guide for AWS EC2 with Docker

Guide này dành cho trạng thái hiện tại của dự án: backend FastAPI ở repo `DominicBE`, frontend React/Vite ở repo `Dominic/chatbot-ui`, frontend sẽ được đưa từ Azure về AWS và chạy cùng EC2.

## 1. Trạng thái hiện tại của dự án

### Đã làm được

- backend đã có `startup.sh` chạy Gunicorn + Uvicorn
- backend đã có health check `/health`, `/health/postgres`, `/health/minio`, `/health/qdrant`
- backend đã có stack local cho `postgres + minio + qdrant` trong `deploy/docker-compose.local-rag.yml`
- frontend đã build tĩnh được bằng `npm run build`
- frontend đã hỗ trợ `VITE_API_BASE_URL=https://api.dominicapp.dev`

### Chưa làm được trước thay đổi này

- chưa có image Docker production cho backend
- chưa có image Docker production cho frontend
- chưa có `docker compose` production gom cả frontend + backend + storage services
- README đang còn một guide EC2 cũ theo `MySQL + systemd`, không còn đúng với kiến trúc hiện tại

### Sau thay đổi này cần dùng những file nào

- backend image: `Dockerfile`
- backend entrypoint: `docker/entrypoint.sh`
- rag-core image: `../rag-core/Dockerfile`
- frontend image: `../Dominic/chatbot-ui/Dockerfile`
- production stack: `deploy/docker-compose.ec2.yml`
- env mẫu cho EC2: `.env.ec2.example`
- Nginx host reverse proxy: `deploy/nginx/dominic-docker-ec2.conf.example`
- service auto-start sau reboot: `deploy/systemd/dominic-docker-compose.service.example`

## 2. Kiến trúc triển khai khuyến nghị

Một EC2 chạy:

- `frontend` container: phục vụ static build của React bằng Nginx
- `backend` container: FastAPI + Gunicorn/Uvicorn
- `rag-core` container: internal FastAPI service cho RAG compute, Qdrant operations, retrieval ranking
- `postgres` container: app DB
- `minio` container: object storage cho file/tài liệu
- `qdrant` container: vector store
- Nginx trên host EC2: reverse proxy + TLS cho domain

Phân domain:

- `https://dominicapp.dev` -> frontend
- `https://www.dominicapp.dev` -> frontend
- `https://api.dominicapp.dev` -> backend

Lưu ý quan trọng về LLM:

- stack Docker này không đóng gói `9router`
- backend hiện dùng OpenAI-compatible provider registry; mặc định cần `NINEROUTER_BASE_URL` và `NINEROUTER_API_KEY`
- nếu `9router` đang chạy trên chính EC2 host, dùng `NINEROUTER_BASE_URL=http://host.docker.internal:20128/v1`
- nếu `9router` nằm ở máy khác, điền URL public/private thật của gateway đó
- model trong `LLM_PROVIDER_CATALOG_JSON` có thể khai báo `contextWindow` và `maxOutputTokens`; giá trị này override fallback `LLM_CONTEXT_WINDOW` và `MAX_OUTPUT_TOKENS`

## 3. Chuẩn bị tài nguyên AWS

### 3.1 Tạo EC2

Khuyến nghị:

- AMI: `Ubuntu Server 24.04 LTS`
- test nhỏ: `t3.medium`
- khuyến nghị production ban đầu cho stack đủ 6 container: `t3.large`
- disk: tối thiểu `40 GB`, nên dùng `60 GB` nếu knowledge base tăng nhanh

### 3.2 Security group

Mở:

- `22` từ IP của bạn
- `80` từ `0.0.0.0/0`
- `443` từ `0.0.0.0/0`

Không cần mở public:

- `5432`
- `6333`
- `6334`
- `9000`
- `9001`
- `8010`
- `8000`
- `8080`

Các cổng này đã được bind nội bộ `127.0.0.1` trong compose, hoặc chỉ `expose` trong Docker network như `rag-core:8010`.

### 3.3 Elastic IP

Nên gắn Elastic IP để domain không đổi khi reboot/recreate instance.

### 3.4 DNS

Trong Route 53 hoặc DNS provider của bạn, tạo:

- A record `dominicapp.dev` -> Elastic IP
- A record `www.dominicapp.dev` -> Elastic IP
- A record `api.dominicapp.dev` -> Elastic IP

Nếu trước đây backend đang nằm ở `dominicapp.dev`, lần này hãy chuyển backend sang `api.dominicapp.dev` và để root domain phục vụ frontend.

## 4. SSH vào EC2 và cài gói hệ thống

Từ Windows PowerShell:

```bash
ssh -i "C:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

Sau khi vào máy:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin nginx certbot python3-certbot-nginx git
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
nginx -v
```

## 5. Clone 2 repo thành sibling directories

```bash
sudo mkdir -p /opt/dominic
sudo chown -R ubuntu:ubuntu /opt/dominic
cd /opt/dominic

git clone https://github.com/admininistrator/DominicBE.git
git clone https://github.com/admininistrator/Dominic.git
```

Sau bước này bạn phải có:

```text
/opt/dominic/DominicBE
/opt/dominic/Dominic
```

Compose production đang giả định đúng layout sibling này để build frontend từ repo `Dominic/chatbot-ui`.

## 6. Tạo file môi trường production

```bash
cd /opt/dominic/DominicBE
cp .env.ec2.example .env.ec2
nano .env.ec2
```

Các biến bắt buộc phải đổi ngay:

- `AUTH_SECRET_KEY`
- `POSTGRES_PASSWORD`
- `MINIO_ROOT_PASSWORD`
- `DATABASE_URL`
- `OBJECT_STORAGE_ACCESS_KEY`
- `OBJECT_STORAGE_SECRET_KEY`
- `NINEROUTER_BASE_URL`
- `NINEROUTER_API_KEY`
- `LLM_DEFAULT_PROVIDER`
- `LLM_DEFAULT_MODEL`
- `CORS_ORIGINS`

Lưu ý đồng bộ:

- `DATABASE_URL` phải dùng cùng credential với `POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB`
- `OBJECT_STORAGE_ACCESS_KEY` và `OBJECT_STORAGE_SECRET_KEY` nên khớp với `MINIO_ROOT_USER/MINIO_ROOT_PASSWORD`
- `FRONTEND_VITE_API_BASE_URL` nên là `https://api.dominicapp.dev`
- `CORS_ORIGINS` nên chứa `https://dominicapp.dev,https://www.dominicapp.dev`

## 7. Dựng stack Docker trên EC2

Vẫn đứng ở `/opt/dominic/DominicBE`:

```bash
docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml up -d --build
```

Kiểm tra trạng thái:

```bash
docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml ps
```

Xem log backend:

```bash
docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml logs -f backend
```

Điều gì xảy ra ở bước này:

- `postgres`, `minio`, `qdrant` lên trước
- `minio-bootstrap` tự tạo bucket `dominic-knowledge`
- `backend` chờ DB, tự chạy `alembic upgrade head`, rồi mới khởi động app
- `frontend` build static assets và phục vụ ở cổng nội bộ `127.0.0.1:8080`

## 8. Kiểm tra nội bộ trước khi public ra internet

Chạy từng lệnh trên EC2:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/postgres
curl http://127.0.0.1:8000/health/minio
curl http://127.0.0.1:8000/health/qdrant
```

Nếu `/health` fail:

- xem `docker compose ... logs backend`
- xem `docker compose ... logs postgres`
- xem `docker compose ... logs minio`
- xem `docker compose ... logs qdrant`

Nếu chat không trả lời nhưng health vẫn xanh:

- kiểm tra `NINEROUTER_BASE_URL`
- kiểm tra `NINEROUTER_API_KEY`
- xác nhận container backend nhìn thấy gateway LLM thật

## 9. Cấu hình Nginx trên host EC2

Copy config mẫu:

```bash
sudo cp /opt/dominic/DominicBE/deploy/nginx/dominic-docker-ec2.conf.example /etc/nginx/sites-available/dominic-docker
sudo nano /etc/nginx/sites-available/dominic-docker
```

Nếu domain của bạn khác, đổi `server_name` cho đúng.

Kích hoạt site:

```bash
sudo ln -sf /etc/nginx/sites-available/dominic-docker /etc/nginx/sites-enabled/dominic-docker
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Kiểm tra public trước khi gắn SSL:

```bash
curl http://dominicapp.dev
curl http://api.dominicapp.dev/health
```

## 10. Gắn HTTPS bằng Let's Encrypt

```bash
sudo certbot --nginx -d dominicapp.dev -d www.dominicapp.dev -d api.dominicapp.dev
```

Sau khi xong, xác thực:

```bash
curl https://dominicapp.dev
curl https://api.dominicapp.dev/health
```

## 11. Tự khởi động stack sau reboot

Copy file service mẫu:

```bash
sudo cp /opt/dominic/DominicBE/deploy/systemd/dominic-docker-compose.service.example /etc/systemd/system/dominic-docker-compose.service
sudo systemctl daemon-reload
sudo systemctl enable dominic-docker-compose.service
sudo systemctl start dominic-docker-compose.service
sudo systemctl status dominic-docker-compose.service
```

Lưu ý:

- service mẫu giả định project nằm ở `/opt/dominic/DominicBE`
- nếu bạn clone ở chỗ khác, sửa lại path trong file service trước khi enable

## 12. Validation checklist sau khi public

### 12.1 Frontend

- mở `https://dominicapp.dev`
- trang phải load được đầy đủ CSS/JS
- login/register form phải render bình thường

### 12.2 Backend

```bash
curl https://api.dominicapp.dev/health
curl https://api.dominicapp.dev/health/postgres
curl https://api.dominicapp.dev/health/minio
curl https://api.dominicapp.dev/health/qdrant
```

### 12.3 Auth

- đăng ký user mới từ UI
- đăng nhập lại
- kiểm tra `/api/auth/me` trả user đúng

### 12.4 Knowledge flow

- upload một file từ UI
- kiểm tra document xuất hiện trong Knowledge panel
- gửi một câu hỏi grounded chat có dùng document đó

### 12.5 MinIO console nếu cần

Vì MinIO console chỉ bind nội bộ, dùng SSH tunnel từ máy local:

```bash
ssh -i "C:\path\to\your-key.pem" -L 9001:127.0.0.1:9001 ubuntu@YOUR_EC2_PUBLIC_IP
```

Sau đó mở:

```text
http://127.0.0.1:9001
```

## 13. Quy trình update sau khi bạn push code lên GitHub

### 13.0 Cách đơn giản nhất: dùng script một lệnh

Repo backend hiện có sẵn script `scripts/deploy_ec2.sh` để gom các bước `git pull + docker compose up -d --build + health check` vào một lệnh.

Lần đầu trên EC2, nhớ cấp quyền chạy:

```bash
cd /opt/dominic/DominicBE
chmod +x scripts/deploy_ec2.sh
```

Sau mỗi lần bạn push code lên GitHub, chỉ cần SSH vào EC2 rồi chạy:

```bash
cd /opt/dominic/DominicBE
./scripts/deploy_ec2.sh
```

Một vài biến thể hữu ích:

```bash
./scripts/deploy_ec2.sh --target backend
./scripts/deploy_ec2.sh --target frontend
./scripts/deploy_ec2.sh --skip-pull
./scripts/deploy_ec2.sh --no-build
```

Script này giả định layout EC2 hiện tại của bạn là:

- `/opt/dominic/DominicBE`
- `/opt/dominic/Dominic/chatbot-ui`

Nếu khác layout, truyền path riêng bằng `--backend-repo` và `--frontend-repo`.

### 13.1 SSH vào EC2

```bash
ssh -i "C:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

### 13.2 Pull cả 2 repo

```bash
cd /opt/dominic/DominicBE
git pull origin main

cd /opt/dominic/Dominic/chatbot-ui
git pull origin main
```

### 13.3 Build lại và rollout lại containers

```bash
cd /opt/dominic/DominicBE
docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml up -d --build
```

### 13.4 Kiểm tra sau update

```bash
docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml ps
docker compose --env-file .env.ec2 -f deploy/docker-compose.ec2.yml logs --tail=100 backend
curl https://api.dominicapp.dev/health
```

## 14. Khi nào cần tách service ra khỏi một EC2 duy nhất

Kiến trúc một EC2 là phù hợp để đưa hệ thống lên nhanh. Khi tải tăng lên, nên cân nhắc:

- chuyển Postgres sang RDS
- chuyển object storage sang S3 thật thay vì MinIO trong EC2
- chuyển vector store sang Qdrant Cloud hoặc node riêng
- đưa frontend ra CloudFront/S3 hoặc Amplify Hosting

Nhưng ở thời điểm hiện tại, stack một EC2 là cách thực tế nhất để đưa cả frontend lẫn backend về AWS nhanh, nhất quán và dễ rollback.
