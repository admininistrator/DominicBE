from __future__ import annotations

import logging

from fastapi import HTTPException, status


GENERIC_INTERNAL_SERVER_ERROR_DETAIL = "An internal server error occurred."


def _log_unexpected_exception(logger: logging.Logger, *, action: str, exc: Exception):
    logger.exception("%s failed error_type=%s", action, type(exc).__name__)


def raise_internal_server_error(
    logger: logging.Logger,
    *,
    action: str,
    exc: Exception,
    detail: str = GENERIC_INTERNAL_SERVER_ERROR_DETAIL,
):
    _log_unexpected_exception(logger, action=action, exc=exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail) from exc


def build_internal_server_error_payload(
    logger: logging.Logger,
    *,
    action: str,
    exc: Exception,
    detail: str = GENERIC_INTERNAL_SERVER_ERROR_DETAIL,
) -> dict[str, int | str]:
    _log_unexpected_exception(logger, action=action, exc=exc)
    return {
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "detail": detail,
    }