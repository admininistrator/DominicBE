#!/bin/bash
PORT="${PORT:-8000}"
echo "Starting gunicorn on port $PORT"
gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --bind "0.0.0.0:$PORT" --timeout 120 --access-logfile - --error-logfile -

