from time import perf_counter

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

connect_args: dict = {}

if settings.sqlalchemy_dialect_name.startswith("mysql"):
    connect_args = {
        "connect_timeout": settings.db_connect_timeout,
        "read_timeout": settings.db_read_timeout,
        "write_timeout": settings.db_write_timeout,
    }

if settings.db_ssl and settings.sqlalchemy_dialect_name.startswith("mysql"):
    ssl_config: dict = {}
    if (settings.db_ssl_ca or "").strip():
        ssl_config["ca"] = settings.db_ssl_ca.strip()
    else:
        ssl_config["ssl_mode"] = "REQUIRED"
    connect_args["ssl"] = ssl_config
elif settings.db_ssl and settings.sqlalchemy_dialect_name.startswith("postgresql"):
    if (settings.db_ssl_ca or "").strip():
        connect_args["sslrootcert"] = settings.db_ssl_ca.strip()
    connect_args["sslmode"] = "require"

engine = create_engine(
    settings.sqlalchemy_database_url,
    pool_pre_ping=True,
    pool_recycle=settings.db_pool_recycle,
    pool_timeout=settings.db_pool_timeout,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def check_database_health() -> dict:
    started_at = perf_counter()
    dialect = settings.sqlalchemy_dialect_name
    dependency = "postgres" if dialect.startswith("postgresql") else dialect or "database"

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "ok": True,
            "dependency": dependency,
            "dialect": dialect,
            "database": settings.db_name,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "dependency": dependency,
            "dialect": dialect,
            "database": settings.db_name,
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "detail": str(exc),
        }
