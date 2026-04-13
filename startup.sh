#!/bin/bash
set -e
PORT="${PORT:-8000}"
echo "=== Starting Dominic Backend ==="
echo "PORT=$PORT"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"
echo "Installed packages:"
pip list 2>/dev/null | grep -iE "fastapi|uvicorn|gunicorn|sqlalchemy|pymysql|anthropic" || true
echo "=== Launching gunicorn ==="
exec gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:$PORT" \
  --timeout 120 \
  --workers 1 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
