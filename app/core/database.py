from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

connect_args: dict = {
    "connect_timeout": settings.db_connect_timeout,
    "read_timeout": settings.db_read_timeout,
    "write_timeout": settings.db_write_timeout,
}

if settings.db_ssl:
    ssl_config: dict = {}
    if (settings.db_ssl_ca or "").strip():
        ssl_config["ca"] = settings.db_ssl_ca.strip()
    else:
        ssl_config["ssl_mode"] = "REQUIRED"
    connect_args["ssl"] = ssl_config

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
