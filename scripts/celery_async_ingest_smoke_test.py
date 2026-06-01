from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

TERMINAL_SUCCESS_STATES = {"completed"}
TERMINAL_FAILURE_STATES = {"failed"}


def _build_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _health_check(client: httpx.Client, base_url: str, health_path: str) -> bool:
    url = _join_url(base_url, health_path)
    print(f"HEALTH: GET {url}")
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        print(f"HEALTH: WARN request failed ({exc}); continuing")
        return False

    if 200 <= response.status_code < 300:
        print(f"HEALTH: OK status={response.status_code}")
        return True

    print(f"HEALTH: WARN status={response.status_code}; continuing")
    return False


def _submit_text(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    title: str,
    text: str,
) -> dict[str, Any]:
    url = _join_url(base_url, "/api/v1/knowledge/documents/ingest")
    payload = {
        "title": title,
        "source_type": "text",
        "raw_text": text,
    }
    print(f"SUBMIT: POST {url}?async_index=true (raw text, {len(text)} chars)")
    response = client.post(url, params={"async_index": "true"}, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def _submit_file(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    file_path: Path,
) -> dict[str, Any]:
    url = _join_url(base_url, "/api/v1/knowledge/documents/upload")
    print(f"SUBMIT: POST {url}?async_index=true (file={file_path})")
    with file_path.open("rb") as handle:
        files = {"file": (file_path.name, handle, "application/octet-stream")}
        response = client.post(url, params={"async_index": "true"}, files=files, headers=headers)
    response.raise_for_status()
    return response.json()


def _poll_job(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    job_id: int,
    timeout_seconds: float,
    interval_seconds: float,
) -> dict[str, Any]:
    url = _join_url(base_url, f"/api/v1/knowledge/jobs/{job_id}")
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None

    while time.monotonic() <= deadline:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        status = str(payload.get("status", ""))
        celery_task_id = payload.get("celery_task_id")
        print(f"POLL: job_id={job_id} status={status} celery_task_id={celery_task_id}")

        if status in TERMINAL_SUCCESS_STATES:
            return payload
        if status in TERMINAL_FAILURE_STATES:
            raise RuntimeError(f"Job failed: {payload.get('error_message') or payload}")

        time.sleep(interval_seconds)

    raise TimeoutError(f"Timed out waiting for job {job_id}; last payload={last_payload}")


def _verify_search(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    query: str,
    document_id: int | None,
) -> dict[str, Any]:
    url = _join_url(base_url, "/api/v1/knowledge/search")
    payload: dict[str, Any] = {"query": query, "top_k": 5}
    if document_id is not None:
        payload["document_id"] = document_id

    print(f"SEARCH: POST {url} query={query!r} document_id={document_id}")
    response = client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    returned = int(result.get("returned") or len(result.get("results") or []))
    if returned <= 0:
        raise RuntimeError(f"Search returned no chunks: {result}")
    print(f"SEARCH: OK returned={returned}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test Celery async knowledge ingestion. Requires a running "
            "Dominic backend with CELERY_ENABLED=true, Redis, and a Celery worker."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DOMINIC_BACKEND_URL", "http://127.0.0.1:8000"),
        help="Backend base URL (default: DOMINIC_BACKEND_URL or http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("DOMINIC_AUTH_TOKEN"),
        help=(
            "Bearer auth token. Defaults to DOMINIC_AUTH_TOKEN. No secrets are "
            "hardcoded; pass a token explicitly for authenticated deployments."
        ),
    )
    parser.add_argument(
        "--health-path",
        default="/health",
        help="Health endpoint path to try before submission (default: /health).",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip backend health check.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a document fixture to upload with async_index=true.",
    )
    parser.add_argument(
        "--text",
        help="Raw text to ingest with async_index=true. Used when --file is omitted.",
    )
    parser.add_argument(
        "--title",
        default="Celery Async Smoke Test",
        help="Title for raw text ingestion (default: Celery Async Smoke Test).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Maximum seconds to wait for job completion (default: 60).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Seconds between job polls (default: 2).",
    )
    parser.add_argument(
        "--search-query",
        help="Query used to verify indexed chunks after completion. Defaults to smoke text/title.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip post-completion knowledge search verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    headers = _build_headers(args.token)

    if args.file and args.text:
        print("FAIL: provide either --file or --text, not both")
        return 1
    if args.file and not args.file.is_file():
        print(f"FAIL: --file does not exist or is not a file: {args.file}")
        return 1

    text = args.text or "Celery async ingestion smoke fixture. Unique token: dominic-celery-smoke."

    if not args.token:
        print("AUTH: WARN no token supplied; request may fail with 401 if backend requires auth")

    try:
        with httpx.Client(timeout=30.0) as client:
            if args.skip_health:
                print("HEALTH: skipped")
            else:
                _health_check(client, base_url, args.health_path)

            if args.file:
                submit_payload = _submit_file(client, base_url, headers, args.file)
            else:
                submit_payload = _submit_text(client, base_url, headers, args.title, text)

            print(f"SUBMIT: response={submit_payload}")
            document_id = submit_payload.get("document_id")
            job_id = submit_payload.get("job_id")
            status = submit_payload.get("status")
            celery_task_id = submit_payload.get("celery_task_id")

            if not job_id:
                raise RuntimeError(f"Submit response did not include job_id: {submit_payload}")
            if status != "queued":
                raise RuntimeError(f"Expected submit status='queued', got {status!r}: {submit_payload}")
            if not celery_task_id:
                print("SUBMIT: WARN response did not include celery_task_id")

            final_job = _poll_job(
                client,
                base_url,
                headers,
                int(job_id),
                args.timeout_seconds,
                args.poll_interval_seconds,
            )

            if args.skip_search:
                print("SEARCH: skipped")
            else:
                query = args.search_query or (args.file.stem if args.file else "dominic-celery-smoke")
                _verify_search(client, base_url, headers, query, int(document_id) if document_id else None)

            print(f"PASS: async ingestion completed job_id={job_id} final_job={final_job}")
            return 0
    except (httpx.HTTPStatusError, httpx.HTTPError, RuntimeError, TimeoutError) as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
