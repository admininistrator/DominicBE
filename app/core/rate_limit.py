from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from threading import Lock
from time import monotonic, time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from app.core.config import Settings
    from sqlalchemy.orm import sessionmaker

from app.models.system_models import RateLimitBucket


@dataclass(frozen=True)
class RateLimitRule:
    name: str
    methods: frozenset[str]
    path: str
    limit: int
    window_seconds: int

    def matches(self, method: str, path: str) -> bool:
        if self.limit <= 0 or self.window_seconds <= 0:
            return False
        if method.upper() not in self.methods:
            return False
        return path in self.all_paths

    @property
    def all_paths(self) -> tuple[str, ...]:
        if self.path.startswith("/api/"):
            return (self.path, f"/api/v1{self.path[4:]}")
        return (self.path,)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int
    retry_after_seconds: int | None = None


class FixedWindowRateLimiter:
    def __init__(self, cleanup_interval_seconds: int = 60, entry_ttl_seconds: int | None = None):
        self._entries: dict[tuple[str, str], deque[float]] = {}
        self._lock = Lock()
        self._cleanup_interval_seconds = max(5, cleanup_interval_seconds)
        self._entry_ttl_seconds = max(self._cleanup_interval_seconds, entry_ttl_seconds or self._cleanup_interval_seconds)
        self._next_cleanup_at = monotonic() + self._cleanup_interval_seconds

    def check(self, *, rule: RateLimitRule, key: str) -> RateLimitDecision:
        now = monotonic()
        cutoff = now - rule.window_seconds
        bucket_key = (rule.name, key)

        with self._lock:
            bucket = self._entries.setdefault(bucket_key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= rule.limit:
                retry_after = max(1, ceil(rule.window_seconds - (now - bucket[0])))
                self._cleanup_if_needed(now)
                return RateLimitDecision(
                    allowed=False,
                    limit=rule.limit,
                    remaining=0,
                    reset_after_seconds=retry_after,
                    retry_after_seconds=retry_after,
                )

            bucket.append(now)
            remaining = max(0, rule.limit - len(bucket))
            reset_after = rule.window_seconds
            if bucket:
                reset_after = max(1, ceil(rule.window_seconds - (now - bucket[0])))
            self._cleanup_if_needed(now)
            return RateLimitDecision(
                allowed=True,
                limit=rule.limit,
                remaining=remaining,
                reset_after_seconds=reset_after,
            )

    def _cleanup_if_needed(self, now: float):
        if now < self._next_cleanup_at:
            return
        cutoff_candidates = []
        for bucket_key, bucket in list(self._entries.items()):
            if not bucket or bucket[-1] <= now - self._entry_ttl_seconds:
                cutoff_candidates.append(bucket_key)
        for bucket_key in cutoff_candidates:
            self._entries.pop(bucket_key, None)
        self._next_cleanup_at = now + self._cleanup_interval_seconds


class DatabaseFixedWindowRateLimiter:
    def __init__(
        self,
        *,
        session_factory: "sessionmaker",
        cleanup_interval_seconds: int = 60,
        entry_ttl_seconds: int | None = None,
    ):
        self._session_factory = session_factory
        self._cleanup_interval_seconds = max(5, cleanup_interval_seconds)
        self._entry_ttl_seconds = max(
            self._cleanup_interval_seconds,
            entry_ttl_seconds or self._cleanup_interval_seconds,
        )
        self._cleanup_lock = Lock()
        self._next_cleanup_at = monotonic() + self._cleanup_interval_seconds

    def check(self, *, rule: RateLimitRule, key: str) -> RateLimitDecision:
        now_epoch = int(time())
        window_start_epoch = now_epoch - (now_epoch % rule.window_seconds)
        reset_after_seconds = max(1, (window_start_epoch + rule.window_seconds) - now_epoch)
        updated_at = datetime.now(UTC).replace(tzinfo=None)

        for attempt in range(2):
            with self._session_factory() as session:
                try:
                    bucket = (
                        session.query(RateLimitBucket)
                        .filter(
                            RateLimitBucket.scope == rule.name,
                            RateLimitBucket.bucket_key == key,
                            RateLimitBucket.window_start_epoch == window_start_epoch,
                        )
                        .with_for_update()
                        .one_or_none()
                    )

                    if bucket is None:
                        bucket = RateLimitBucket(
                            scope=rule.name,
                            bucket_key=key,
                            window_start_epoch=window_start_epoch,
                            request_count=1,
                            updated_at=updated_at,
                        )
                        session.add(bucket)
                        session.commit()
                        self._cleanup_if_needed(now_epoch)
                        return RateLimitDecision(
                            allowed=True,
                            limit=rule.limit,
                            remaining=max(0, rule.limit - 1),
                            reset_after_seconds=reset_after_seconds,
                        )

                    if int(bucket.request_count or 0) >= rule.limit:
                        session.rollback()
                        self._cleanup_if_needed(now_epoch)
                        return RateLimitDecision(
                            allowed=False,
                            limit=rule.limit,
                            remaining=0,
                            reset_after_seconds=reset_after_seconds,
                            retry_after_seconds=reset_after_seconds,
                        )

                    bucket.request_count = int(bucket.request_count or 0) + 1
                    bucket.updated_at = updated_at
                    session.commit()
                    self._cleanup_if_needed(now_epoch)
                    return RateLimitDecision(
                        allowed=True,
                        limit=rule.limit,
                        remaining=max(0, rule.limit - bucket.request_count),
                        reset_after_seconds=reset_after_seconds,
                    )
                except IntegrityError:
                    session.rollback()
                    if attempt == 0:
                        continue
                    raise

        raise RuntimeError("Failed to evaluate rate limit after retry.")

    def _cleanup_if_needed(self, now_epoch: int):
        now = monotonic()
        if now < self._next_cleanup_at:
            return
        if not self._cleanup_lock.acquire(blocking=False):
            return
        try:
            if now < self._next_cleanup_at:
                return
            stale_before_epoch = now_epoch - self._entry_ttl_seconds
            with self._session_factory() as session:
                session.query(RateLimitBucket).filter(
                    RateLimitBucket.window_start_epoch < stale_before_epoch,
                ).delete(synchronize_session=False)
                session.commit()
            self._next_cleanup_at = now + self._cleanup_interval_seconds
        finally:
            self._cleanup_lock.release()


def build_default_rate_limit_rules(settings: "Settings") -> list[RateLimitRule]:
    return [
        RateLimitRule(
            name="auth.register",
            methods=frozenset({"POST"}),
            path="/api/auth/register",
            limit=settings.rate_limit_auth_register_requests,
            window_seconds=settings.rate_limit_auth_register_window_seconds,
        ),
        RateLimitRule(
            name="auth.login",
            methods=frozenset({"POST"}),
            path="/api/auth/login",
            limit=settings.rate_limit_auth_login_requests,
            window_seconds=settings.rate_limit_auth_login_window_seconds,
        ),
        RateLimitRule(
            name="auth.reset_password",
            methods=frozenset({"POST"}),
            path="/api/auth/reset-password",
            limit=settings.rate_limit_auth_reset_password_requests,
            window_seconds=settings.rate_limit_auth_reset_password_window_seconds,
        ),
        RateLimitRule(
            name="chat.send",
            methods=frozenset({"POST"}),
            path="/api/chat/",
            limit=settings.rate_limit_chat_requests,
            window_seconds=settings.rate_limit_chat_window_seconds,
        ),
        RateLimitRule(
            name="chat.stream",
            methods=frozenset({"POST"}),
            path="/api/chat/stream",
            limit=settings.rate_limit_chat_stream_requests,
            window_seconds=settings.rate_limit_chat_stream_window_seconds,
        ),
        RateLimitRule(
            name="knowledge.upload",
            methods=frozenset({"POST"}),
            path="/api/knowledge/documents/upload",
            limit=settings.rate_limit_knowledge_upload_requests,
            window_seconds=settings.rate_limit_knowledge_upload_window_seconds,
        ),
        RateLimitRule(
            name="knowledge.ingest",
            methods=frozenset({"POST"}),
            path="/api/knowledge/documents/ingest",
            limit=settings.rate_limit_knowledge_ingest_requests,
            window_seconds=settings.rate_limit_knowledge_ingest_window_seconds,
        ),
    ]


def find_matching_rule(rules: list[RateLimitRule], *, method: str, path: str) -> RateLimitRule | None:
    for rule in rules:
        if rule.matches(method, path):
            return rule
    return None


def build_rate_limiter(
    settings: "Settings",
    *,
    session_factory: "sessionmaker" | None = None,
):
    rules = build_default_rate_limit_rules(settings)
    entry_ttl_seconds = max(
        [rule.window_seconds for rule in rules if rule.limit > 0],
        default=settings.rate_limit_cleanup_interval_seconds,
    )
    if settings.rate_limit_store_backend == "database":
        if session_factory is None:
            raise ValueError("session_factory is required when RATE_LIMIT_STORE_BACKEND=database")
        return DatabaseFixedWindowRateLimiter(
            session_factory=session_factory,
            cleanup_interval_seconds=settings.rate_limit_cleanup_interval_seconds,
            entry_ttl_seconds=entry_ttl_seconds,
        )
    return FixedWindowRateLimiter(
        cleanup_interval_seconds=settings.rate_limit_cleanup_interval_seconds,
        entry_ttl_seconds=entry_ttl_seconds,
    )


def get_client_ip(request: Request, *, trust_proxy_headers: bool = True) -> str:
    if trust_proxy_headers:
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            first_hop = forwarded_for.split(",", 1)[0].strip()
            if first_hop:
                return first_hop
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
    return getattr(request.client, "host", None) or "unknown"


def attach_rate_limit_headers(response, decision: RateLimitDecision):
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    response.headers["X-RateLimit-Reset"] = str(decision.reset_after_seconds)
    if decision.retry_after_seconds is not None:
        response.headers["Retry-After"] = str(decision.retry_after_seconds)
    return response


def build_rate_limited_response(*, rule: RateLimitRule, decision: RateLimitDecision) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded for '{rule.name}'. Try again later.",
            "error": "rate_limit_exceeded",
            "scope": rule.name,
        },
    )
    return attach_rate_limit_headers(response, decision)