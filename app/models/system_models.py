from sqlalchemy import BigInteger, Column, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from app.core.database import Base


class RateLimitBucket(Base):
    __tablename__ = "rate_limit_buckets"
    __table_args__ = (
        UniqueConstraint(
            "scope",
            "bucket_key",
            "window_start_epoch",
            name="uq_rate_limit_buckets_scope_key_window",
        ),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scope = Column(String(128), nullable=False, index=True)
    bucket_key = Column(String(255), nullable=False, index=True)
    window_start_epoch = Column(BigInteger, nullable=False, index=True)
    request_count = Column(Integer, nullable=False, server_default="1")
    created_at = Column(DateTime, server_default=func.now(), nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True, index=True)