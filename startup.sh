#!/bin/bash
set -e

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"

echo "=== Starting Dominic Backend ==="
echo "HOST=$HOST"
echo "PORT=$PORT"
echo "WEB_CONCURRENCY=$WEB_CONCURRENCY"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"
echo "Installed packages:"
pip list 2>/dev/null | grep -iE "fastapi|uvicorn|gunicorn|sqlalchemy|pymysql|anthropic|alembic|passlib" || true

echo "=== Launching gunicorn ==="
exec gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "$HOST:$PORT" \
  --timeout 120 \
  --workers "$WEB_CONCURRENCY" \
  --access-logfile - \
  --error-logfile - \
  --log-level info
