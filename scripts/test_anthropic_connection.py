import os
import sys
from pathlib import Path

from anthropic import APIConnectionError, APIStatusError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def chain(exc: BaseException | None) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{current.__class__.__name__}: {str(current).strip() or repr(current)}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def main() -> int:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    model = (os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-0").strip()
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip() or "https://api.anthropic.com"
    force_ipv4 = (os.getenv("ANTHROPIC_FORCE_IPV4") or "").strip().lower() in {"1", "true", "yes", "on"}

    print("=== Dominic Anthropic diagnostic ===")
    print(f"python={sys.version.split()[0]}")
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"force_ipv4={force_ipv4}")
    print(f"api_key_set={bool(api_key)}")

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is missing")
        return 2

    from app.services.chat_service import _get_client

    try:
        client = _get_client()
        print("client_init=ok")

        try:
            count_tokens = getattr(client.messages, "count_tokens", None)
            if callable(count_tokens):
                token_info = count_tokens(
                    model=model,
                    messages=[{"role": "user", "content": "Ping"}],
                )
                print(f"count_tokens=ok input_tokens={int(getattr(token_info, 'input_tokens', 0) or 0)}")
            else:
                print("count_tokens=skip sdk_does_not_expose_count_tokens")
        except Exception as exc:
            print(f"count_tokens=fail {chain(exc)}")

        response = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Say hi in one short sentence."}],
        )
        text = response.content[0].text.strip() if response.content else ""
        print(f"messages_create=ok text={text}")
        return 0
    except APIStatusError as exc:
        print(f"api_status_error={getattr(exc, 'status_code', '?')} chain={chain(exc)}")
        return 3
    except APIConnectionError as exc:
        print(f"api_connection_error chain={chain(exc)}")
        return 4
    except Exception as exc:
        print(f"unexpected_error chain={chain(exc)}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())


