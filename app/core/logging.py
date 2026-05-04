from __future__ import annotations

import logging
import sys
from contextvars import ContextVar


_LOG_CONTEXT: ContextVar[dict[str, str]] = ContextVar("dominic_log_context", default={})
_DEFAULT_CONTEXT = {
    "request_id": "-",
    "username": "-",
    "client_ip": "-",
    "http_method": "-",
    "http_path": "-",
}


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = {**_DEFAULT_CONTEXT, **(_LOG_CONTEXT.get() or {})}
        record.request_id = context["request_id"]
        record.username = context["username"]
        record.client_ip = context["client_ip"]
        record.http_method = context["http_method"]
        record.http_path = context["http_path"]
        return True


def configure_logging(level: int = logging.INFO):
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s level=%(levelname)s logger=%(name)s request_id=%(request_id)s "
        "username=%(username)s ip=%(client_ip)s method=%(http_method)s path=%(http_path)s "
        "message=%(message)s"
    )

    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler(sys.stdout))

    for handler in root_logger.handlers:
        if not any(isinstance(existing_filter, RequestContextFilter) for existing_filter in handler.filters):
            handler.addFilter(RequestContextFilter())
        handler.setFormatter(formatter)

    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_log_context(**kwargs):
    current_context = dict(_LOG_CONTEXT.get() or {})
    for key, value in kwargs.items():
        if value in (None, ""):
            current_context.pop(key, None)
        else:
            current_context[key] = str(value)
    _LOG_CONTEXT.set(current_context)


def clear_log_context():
    _LOG_CONTEXT.set({})


def get_log_context() -> dict[str, str]:
    return {**_DEFAULT_CONTEXT, **(_LOG_CONTEXT.get() or {})}


def restore_log_context(context: dict[str, str] | None):
    restored_context = dict(context or {})
    for key in _DEFAULT_CONTEXT:
        if restored_context.get(key) in (None, "-"):
            restored_context.pop(key, None)
    _LOG_CONTEXT.set(restored_context)