"""Redis broker smoke test for Celery deployments.

Reads CELERY_BROKER_URL by default, pings Redis with redis-py, prints a
human-readable PASS/FAIL line, and exits 0/1.
"""
from __future__ import annotations

import argparse
import os
import sys

import redis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ping the Redis broker configured by CELERY_BROKER_URL.")
    parser.add_argument(
        "--url",
        default=None,
        help="Redis URL override. Defaults to CELERY_BROKER_URL from the environment.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Socket connect/read timeout in seconds (default: 5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    broker_url = args.url or os.getenv("CELERY_BROKER_URL")
    if not broker_url:
        print("FAIL Redis smoke: CELERY_BROKER_URL is not set and --url was not provided.")
        return 1

    try:
        client = redis.Redis.from_url(
            broker_url,
            socket_connect_timeout=args.timeout,
            socket_timeout=args.timeout,
        )
        if client.ping() is not True:
            print("FAIL Redis smoke: PING returned a non-true response.")
            return 1
    except Exception as exc:  # noqa: BLE001 - smoke scripts should report any runtime failure clearly.
        print(f"FAIL Redis smoke: {type(exc).__name__}: {exc}")
        return 1

    print("PASS Redis smoke: broker responded to PING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
