from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = "Please stream a short deterministic greeting."


def _parse_sse_event(lines: list[str]) -> dict[str, Any] | None:
    event_name: str | None = None
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ").strip()
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))

    if event_name is None:
        return None

    payload: Any = None
    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = raw_data

    return {"event": event_name, "data": payload}


def _iter_sse_events(response: requests.Response):
    pending_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip("\r")
        if line == "":
            event = _parse_sse_event(pending_lines)
            pending_lines = []
            if event is not None:
                yield event
            continue
        pending_lines.append(line)

    if pending_lines:
        event = _parse_sse_event(pending_lines)
        if event is not None:
            yield event


def run_streaming_smoke_test(
    *,
    base_url: str,
    token: str,
    session_id: int,
    username: str | None,
    message: str,
    timeout: float,
) -> int:
    endpoint = f"{base_url.rstrip('/')}/api/v1/chat/stream"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "session_id": session_id,
        "message": message,
        "knowledge_document_id": None,
        "use_web_search": False,
        "images": [],
        "image_media_types": [],
    }
    if username:
        payload["username"] = username

    started_at = time.monotonic()
    time_to_first_delta: float | None = None
    events: list[dict[str, Any]] = []

    with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            raise AssertionError(f"Expected text/event-stream response, got {content_type!r}")

        for event in _iter_sse_events(response):
            events.append(event)
            event_name = event["event"]
            print(f"event={event_name} data={event['data']}")

            if event_name == "delta" and time_to_first_delta is None:
                data = event.get("data") or {}
                if isinstance(data, dict) and data.get("text"):
                    time_to_first_delta = time.monotonic() - started_at

            if event_name in {"final", "error"}:
                break

    total_stream_time = time.monotonic() - started_at
    event_names = [event["event"] for event in events]

    if not events:
        raise AssertionError("No SSE events received")
    if event_names[0] != "start":
        raise AssertionError(f"First SSE event must be start, got {event_names[0]!r}")
    if "delta" not in event_names:
        raise AssertionError("No delta event with streamed text was received")
    if event_names[-1] != "final":
        raise AssertionError(f"Final SSE event must be final, got {event_names[-1]!r}")

    final_data = events[-1].get("data") or {}
    if not isinstance(final_data, dict) or final_data.get("success") is not True:
        raise AssertionError(f"Final event did not report success: {final_data!r}")

    if time_to_first_delta is None:
        raise AssertionError("Unable to measure time_to_first_delta because no non-empty delta was received")

    print("\nStreaming smoke test PASS")
    print(f"time_to_first_delta={time_to_first_delta:.3f}s")
    print(f"total_stream_time={total_stream_time:.3f}s")
    print(f"event_sequence={event_names}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual smoke test for POST /api/v1/chat/stream against a live backend.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL without /api/v1")
    parser.add_argument("--token", required=True, help="Bearer access token for a live backend user")
    parser.add_argument("--session-id", required=True, type=int, help="Existing chat session ID for the token owner")
    parser.add_argument("--username", default=None, help="Optional username if the backend requires explicit username matching")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Message to send to the streaming endpoint")
    parser.add_argument("--timeout", default=120.0, type=float, help="requests timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_streaming_smoke_test(
        base_url=args.base_url,
        token=args.token,
        session_id=args.session_id,
        username=args.username,
        message=args.message,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
