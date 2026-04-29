#!/bin/sh
set -eu

if [ "${WAIT_FOR_DATABASE:-1}" = "1" ]; then
  python - <<'PY'
import os
import time

from sqlalchemy import create_engine, text

database_url = (os.environ.get("DATABASE_URL") or "").strip()
if not database_url:
    raise SystemExit(0)

retries = int(os.environ.get("DB_WAIT_RETRIES", "30"))
sleep_seconds = float(os.environ.get("DB_WAIT_SLEEP_SECONDS", "2"))

for attempt in range(1, retries + 1):
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database connectivity confirmed.")
        break
    except Exception as exc:
        if attempt == retries:
            raise
        print(f"Waiting for database ({attempt}/{retries}): {exc}")
        time.sleep(sleep_seconds)
PY
fi

if [ "${RUN_MIGRATIONS:-1}" = "1" ]; then
  echo "=== Running Alembic migrations ==="
  alembic upgrade head
fi

exec ./startup.sh