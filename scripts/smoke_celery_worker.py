"""Celery worker smoke test.

Uses the configured app.worker.celery_app.celery_app control channel to verify
that at least one worker responds to inspect().ping().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.worker.celery_app import celery_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ping Celery workers through the configured Celery app.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Celery inspect timeout in seconds (default: 5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inspector = celery_app.control.inspect(timeout=args.timeout)
        responses = inspector.ping() if inspector is not None else None
    except Exception as exc:  # noqa: BLE001 - smoke scripts should report any runtime failure clearly.
        print(f"FAIL Celery worker smoke: {type(exc).__name__}: {exc}")
        return 1

    if not responses:
        print("FAIL Celery worker smoke: no workers responded to inspect().ping().")
        return 1

    worker_names = ", ".join(sorted(str(name) for name in responses.keys()))
    print(f"PASS Celery worker smoke: {len(responses)} worker(s) responded: {worker_names}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
