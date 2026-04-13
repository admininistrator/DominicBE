# DominicBE deployment guide (AWS EC2 - Singapore)

This backend is a FastAPI app using:
- FastAPI + Gunicorn/Uvicorn
- MySQL
- Anthropic API

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
ENABLE_DEBUG_ENV=false
```

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
```

Expected:

```json
{"service":"Dominic Backend","status":"running"}
```

and

```json
{"ok":true}
```

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
Passwords are currently compared as plain text in this project.

Open MySQL:

```bash
mysql -u dominic -p
```

Then:

```sql
USE chatbot_db;
INSERT INTO users (username, password, max_tokens_per_day)
VALUES ('test_user', '123456', 10000);
```

If the user already exists:

```sql
UPDATE users
SET password = '123456'
WHERE username = 'test_user';
```

---

## 18. Validation checklist after deployment

Run these checks in order.

### 18.1 Backend local on server

```bash
curl http://127.0.0.1:8000/health
```

### 18.2 Backend through Nginx

```bash
curl http://YOUR_DOMAIN_OR_EC2_IP/health
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
- login
- create session
- send a prompt

If login works but sending prompt fails, inspect:
- `sudo journalctl -u dominic -f`
- Anthropic key/model
- outbound network from EC2

### 18.7 Direct Anthropic diagnostic on EC2

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
```

If using Nginx publicly:

```bash
curl http://YOUR_DOMAIN_OR_EC2_IP/health
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

