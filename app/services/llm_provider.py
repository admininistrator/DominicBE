"""Unified LLM provider layer using LiteLLM over 9router.

Chat and vision requests are normalized onto the OpenAI-compatible 9router
gateway so the rest of the application can switch among a small allowlist of
user-facing model names without changing call sites.

Image processing pipeline (applied automatically):
    1. Resize  – longest side ≤ LLM_IMAGE_MAX_DIMENSION (default 1 568 px)
    2. Format  – RGBA/palette PNG → RGB JPEG (smaller payload)
    3. OCR     – text-heavy images (screenshots, scanned docs) are identified
                 via Tesseract confidence or pixel-statistics heuristic.
                 When OCR text is extracted, the image bytes are *not* sent to
                 the vision model → significant token savings.

Prompt caching hooks are retained for backward compatibility, but the current
9router flow does not add provider-specific cache directives.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from collections.abc import Iterator
from typing import Any

import litellm
import requests
from litellm import ModelResponse
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    RateLimitError,
    ServiceUnavailableError,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Silence verbose litellm logging ──────────────────────────────────────────
litellm.suppress_debug_info = True
litellm.set_verbose = False


# ── Model resolution ──────────────────────────────────────────────────────────

DEFAULT_9ROUTER_MODEL = "gpt-5.4"

MODEL_TARGET_ALIASES = {
    "deepseek-v4-pro": "deepseek_v4_pro_target",
}

KILOCODE_REASONING_MODELS = {
    "openai/kc/moonshotai/kimi-k2.6",
    "openai/kc/qwen/qwen3.6-plus",
}

KILOCODE_REASONING_EFFORT_MAP = {
    "instant": "none",
    "thinking": "minimal",
}


def list_supported_chat_models() -> list[str]:
    configured = settings.supported_chat_models
    return configured or [DEFAULT_9ROUTER_MODEL]


def _normalize_9router_model_name(model_name: str | None) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("openai/"):
        normalized = normalized.split("/", 1)[1]
    if normalized.startswith("gh/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def _validate_9router_model_name(model_name: str | None) -> str:
    normalized = _normalize_9router_model_name(model_name)
    if not normalized:
        return ""

    supported = list_supported_chat_models()
    if normalized not in supported:
        raise RuntimeError(
            "Unsupported chat model '%s'. Supported models: %s"
            % (normalized, ", ".join(supported))
        )
    return normalized


def _resolve_configured_model_name(model_name: str | None, *, source: str) -> str:
    normalized = _normalize_9router_model_name(model_name)
    if not normalized:
        return ""

    supported = list_supported_chat_models()
    if normalized not in supported:
        logger.warning(
            "Ignoring unsupported configured model from %s: %s. Falling back to %s.",
            source,
            normalized,
            DEFAULT_9ROUTER_MODEL,
        )
        return ""
    return normalized


def _to_litellm_model(model_name: str) -> str:
    normalized = _normalize_9router_model_name(model_name)
    if not normalized:
        return ""

    target_setting_name = MODEL_TARGET_ALIASES.get(normalized)
    if target_setting_name:
        configured_target = getattr(settings, target_setting_name, "")
        target = (configured_target or "").strip()
        if target:
            return f"openai/{target}"

    if "/" in normalized:
        return f"openai/{normalized}"

    return f"openai/gh/{normalized}"


def _normalize_reasoning_effort(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized or None


def _provider_reasoning_effort(model_str: str, value: str | None) -> str | None:
    normalized = _normalize_reasoning_effort(value)
    if not normalized:
        return None
    if model_str in KILOCODE_REASONING_MODELS:
        return KILOCODE_REASONING_EFFORT_MAP.get(normalized)
    return normalized


def _supports_reasoning_effort(model_str: str) -> bool:
    return model_str.startswith("openai/gh/gpt-") or model_str in {
        "openai/gh/claude-sonnet-4.6",
        "openai/gh/claude-opus-4.6",
    } or model_str in KILOCODE_REASONING_MODELS

def resolve_model(model: str | None = None) -> str:
    """Return the full LiteLLM model string for the 9router-backed chat model."""
    explicit = _validate_9router_model_name(model)
    if explicit:
        return _to_litellm_model(explicit)

    configured = _resolve_configured_model_name(settings.llm_model, source="LLM_MODEL")
    if configured:
        return _to_litellm_model(configured)

    configured = _resolve_configured_model_name(
        settings.github_copilot_model_name,
        source="MODEL_GITHUB_COPILOT",
    )
    if configured:
        return _to_litellm_model(configured)

    return _to_litellm_model(DEFAULT_9ROUTER_MODEL)


def resolve_vision_model(model: str | None = None) -> str:
    """Return vision-capable model string."""
    explicit = _validate_9router_model_name(model)
    if explicit:
        return _to_litellm_model(explicit)

    configured = _resolve_configured_model_name(settings.llm_vision_model, source="LLM_VISION_MODEL")
    if configured:
        return _to_litellm_model(configured)

    return resolve_model()


# ── API key / extra kwargs per provider ──────────────────────────────────────

def _provider_name(model_str: str) -> str:
    """Extract provider name from a LiteLLM model string."""
    return model_str.split("/")[0].lower() if "/" in model_str else "openai"


def _provider_kwargs(model_str: str) -> dict[str, Any]:
    """Inject 9router API credentials into the LiteLLM call."""
    if not model_str.startswith("openai/"):
        raise RuntimeError(
            "Only 9router-backed 'openai/*' chat models are supported in this project."
        )

    api_key = (settings.github_copilot_api_key or "").strip() or (settings.openai_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "GITHUB_COPILOT_API_KEY is not set. Configure it in .env or environment variables."
        )

    base_url = (settings.ninerouter_base_url or "").strip() or (settings.openai_base_url or "").strip()
    if not base_url:
        raise RuntimeError(
            "NINEROUTER_BASE_URL is not set. Configure it in .env or environment variables."
        )

    return {
        "api_key": api_key,
        "api_base": base_url,
    }


def _uses_direct_9router_http(model_str: str) -> bool:
    return model_str.startswith("openai/gh/gemini-")


def _uses_direct_9router_chat_http(model_str: str) -> bool:
    return model_str == "openai/kc/minimax/minimax-m2.7"


def _uses_direct_9router_reasoning_chat_http(model_str: str, reasoning_effort: str | None) -> bool:
    return model_str in KILOCODE_REASONING_MODELS and bool(reasoning_effort)


def _extract_json_from_9router_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        pass

    text = response.text or ""
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate == "data: [DONE]":
            continue
        if candidate.startswith("data: "):
            candidate = candidate[6:].strip()
        try:
            return json.loads(candidate)
        except ValueError:
            continue

    raise ValueError("9router response did not contain a valid JSON payload")


def _extract_text_for_token_estimate(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text") or ""))
                continue
            if item_type == "input_text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""


def _estimate_token_count_from_text(text: str) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // max(1, settings.token_estimate_chars_per_token))


def _estimate_token_count_from_messages(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(_extract_text_for_token_estimate(message.get("content")))
    if total_chars <= 0:
        return 0
    return max(1, total_chars // max(1, settings.token_estimate_chars_per_token))


def _complete_via_9router_http(
    *,
    model_str: str,
    call_messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    base_url = (extra_kwargs.get("api_base") or "").rstrip("/")
    api_key = (extra_kwargs.get("api_key") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("Missing 9router credentials for direct Gemini call")

    payload: dict[str, Any] = {
        "model": model_str.split("openai/", 1)[1],
        "input": call_messages,
        "stream": False,
    }
    if max_tokens:
        payload["max_output_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        f"{base_url}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if response.status_code == 401:
        raise LLMError(401, "Xác thực thất bại với provider 'openai'. Kiểm tra API key.")
    if response.status_code == 429:
        raise LLMError(429, "Provider đang giới hạn tốc độ. Vui lòng thử lại sau.")
    if response.status_code >= 500:
        raise LLMError(503, "Provider 'openai' tạm thời không khả dụng. Thử lại sau.")
    if response.status_code >= 400:
        detail = response.text.strip()[:200] or "Yêu cầu không hợp lệ"
        raise LLMError(400, f"Yêu cầu không hợp lệ: {detail}")

    data = _extract_json_from_9router_response(response)
    choices = data.get("choices") or []
    message = choices[0].get("message") if choices else {}
    usage = data.get("usage") or {}
    text = message.get("content") or ""
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    if input_tokens <= 0:
        input_tokens = _estimate_token_count_from_messages(call_messages)
    if output_tokens <= 0:
        output_tokens = _estimate_token_count_from_text(text)
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model": data.get("model") or model_str,
    }


def _complete_via_9router_chat_http(
    *,
    model_str: str,
    call_messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    base_url = (extra_kwargs.get("api_base") or "").rstrip("/")
    api_key = (extra_kwargs.get("api_key") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("Missing 9router credentials for direct chat call")

    payload: dict[str, Any] = {
        "model": model_str.split("openai/", 1)[1],
        "messages": call_messages,
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if response.status_code == 401:
        raise LLMError(401, "Xác thực thất bại với provider 'openai'. Kiểm tra API key.")
    if response.status_code == 429:
        raise LLMError(429, "Provider đang giới hạn tốc độ. Vui lòng thử lại sau.")
    if response.status_code >= 500:
        raise LLMError(503, "Provider 'openai' tạm thời không khả dụng. Thử lại sau.")
    if response.status_code >= 400:
        detail = response.text.strip()[:200] or "Yêu cầu không hợp lệ"
        raise LLMError(400, f"Yêu cầu không hợp lệ: {detail}")

    data = _extract_json_from_9router_response(response)
    choices = data.get("choices") or []
    message = choices[0].get("message") if choices else {}
    usage = data.get("usage") or {}
    text = message.get("content") or ""
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    if input_tokens <= 0:
        input_tokens = _estimate_token_count_from_messages(call_messages)
    if output_tokens <= 0:
        output_tokens = _estimate_token_count_from_text(text)
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model": data.get("model") or model_str,
    }


# ── Image helpers ─────────────────────────────────────────────────────────────

def _make_image_block(image_data: str | bytes, media_type: str = "image/jpeg") -> dict:
    """Build an OpenAI-compatible image_url content block.

    ``image_data`` may be:
    - bytes → encoded to base64 data-URI
    - str already starting with "data:" → used as-is
    - str (raw base64) → wrapped in data-URI
    """
    if isinstance(image_data, bytes):
        b64 = base64.b64encode(image_data).decode("ascii")
        url = f"data:{media_type};base64,{b64}"
    elif isinstance(image_data, str) and image_data.startswith("data:"):
        url = image_data
    else:
        # Assume raw base64 string
        url = f"data:{media_type};base64,{image_data}"

    return {"type": "image_url", "image_url": {"url": url}}


def _guess_media_type(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "image/jpeg"


def _preprocess_image(image_data: str | bytes, media_type: str) -> tuple[str | bytes, str, str | None]:
    """Run the image preprocessing pipeline.

    Returns:
        (processed_image, final_media_type, ocr_text_or_None)
        When ocr_text is not None the caller should use text instead of image.
    """
    if not settings.llm_image_resize_enabled and not settings.llm_image_ocr_enabled:
        return image_data, media_type, None

    # Resolve bytes
    if isinstance(image_data, str):
        if image_data.startswith("data:"):
            # Extract base64 part
            try:
                _, b64part = image_data.split(",", 1)
                raw_bytes = base64.b64decode(b64part)
            except Exception:
                return image_data, media_type, None
        else:
            try:
                raw_bytes = base64.b64decode(image_data)
            except Exception:
                return image_data, media_type, None
    else:
        raw_bytes = image_data

    try:
        from app.services.image_processor import preprocess_for_llm
        result = preprocess_for_llm(
            raw_bytes,
            media_type,
            max_dimension=settings.llm_image_max_dimension if settings.llm_image_resize_enabled else 99999,
            ocr_enabled=settings.llm_image_ocr_enabled,
            ocr_confidence_threshold=settings.llm_image_ocr_confidence_threshold,
        )

        if result.notes:
            logger.debug("ImagePreprocess: %s", " | ".join(result.notes))

        # OCR succeeded with meaningful text → skip vision
        if not result.use_vision and result.ocr_text.strip():
            return result.image_bytes, result.media_type, result.ocr_text.strip()

        return result.image_bytes, result.media_type, None

    except Exception as exc:
        logger.warning("Image preprocessing failed (non-fatal): %s", exc)
        return image_data, media_type, None


# ── Prompt caching hook (currently a no-op for 9router) ─────────────────────

def _apply_prompt_caching(
    call_messages: list[dict],
    system: str | None,
    model_str: str,
) -> tuple[list[dict], str | None]:
    del model_str
    return call_messages, system


# ── Core completion ───────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised for mapped provider errors with a user-facing message and HTTP status."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def complete(
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    model: str | None = None,
    images: list[str | bytes] | None = None,
    image_media_types: list[str] | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Call the configured LLM and return a normalized response dict.

    Images are automatically preprocessed:
      - Resized to ≤ LLM_IMAGE_MAX_DIMENSION on longest side
      - Converted to JPEG when possible
      - OCR-extracted text substitutes the image when Tesseract detects
        a text-heavy image (saves vision token cost entirely)

    Prompt caching is applied automatically for Anthropic when the system
    prompt is ≥ LLM_PROMPT_CACHING_MIN_CHARS.

    Returns:
        {"text": str, "input_tokens": int, "output_tokens": int, "model": str,
         "cache_read_tokens": int, "cache_write_tokens": int}
    """
    model_str = resolve_model(model) if not images else resolve_vision_model(model)
    extra_kwargs = _provider_kwargs(model_str)

    # ── Image preprocessing ───────────────────────────────────────────────
    processed_images: list[str | bytes] = []
    processed_media_types: list[str] = []
    ocr_texts: list[str] = []

    for idx, img in enumerate(images or []):
        mt = (image_media_types or [])[idx] if idx < len(image_media_types or []) else "image/jpeg"
        p_img, p_mt, ocr_text = _preprocess_image(img, mt)
        if ocr_text:
            ocr_texts.append(ocr_text)
            logger.info("Image %d: OCR extracted %d chars, skipping vision", idx + 1, len(ocr_text))
        else:
            processed_images.append(p_img)
            processed_media_types.append(p_mt)

    # Inject OCR texts into system prompt so the model treats them as
    # pre-processed context, NOT as user-pasted content.
    # This prevents the model from saying "I cannot extract text from images".
    call_messages = list(messages)
    effective_system = system
    if ocr_texts:
        ocr_blocks = "\n\n".join(
            f"--- Ảnh {i+1} ---\n{t}"
            for i, t in enumerate(ocr_texts)
        )
        ocr_system_note = (
            "Hệ thống đã tự động trích xuất văn bản từ ảnh đính kèm bằng OCR. "
            "Hãy sử dụng nội dung dưới đây để trả lời câu hỏi của người dùng một cách tự nhiên, "
            "như thể bạn đã phân tích ảnh trực tiếp. "
            "Không đề cập đến việc OCR hay không thể xem ảnh.\n\n"
            f"Nội dung trích xuất từ ảnh:\n{ocr_blocks}"
        )
        effective_system = (effective_system + "\n\n" + ocr_system_note) if effective_system else ocr_system_note

    # ── Inject vision images ──────────────────────────────────────────────
    if processed_images:
        call_messages = _inject_images(call_messages, processed_images, processed_media_types)

    # ── Prompt caching ────────────────────────────────────────────────────
    call_messages, system_after_cache = _apply_prompt_caching(call_messages, effective_system, model_str)

    # Add system as plain message if caching didn't consume it
    if system_after_cache:
        call_messages = [{"role": "system", "content": system_after_cache}] + call_messages

    # ── Build call kwargs ─────────────────────────────────────────────────
    call_kwargs: dict[str, Any] = {
        "model": model_str,
        "messages": call_messages,
        "max_tokens": max_tokens,
        **extra_kwargs,
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature

    effective_reasoning_effort = _normalize_reasoning_effort(reasoning_effort) or _normalize_reasoning_effort(settings.llm_reasoning_effort)
    provider_reasoning_effort = _provider_reasoning_effort(model_str, effective_reasoning_effort)
    if provider_reasoning_effort and _supports_reasoning_effort(model_str):
        call_kwargs["reasoning_effort"] = provider_reasoning_effort
        if model_str in {"openai/gh/claude-sonnet-4.6", "openai/gh/claude-opus-4.6"} | KILOCODE_REASONING_MODELS:
            call_kwargs["allowed_openai_params"] = ["reasoning_effort"]
    elif effective_reasoning_effort:
        logger.info(
            "Ignoring reasoning_effort=%s for unsupported model=%s",
            effective_reasoning_effort,
            model_str,
        )

    logger.debug(
        "LiteLLM call model=%s messages=%d vision_imgs=%d ocr_imgs=%d max_tokens=%d",
        model_str, len(call_messages), len(processed_images), len(ocr_texts), max_tokens,
    )

    if _uses_direct_9router_http(model_str):
        return _complete_via_9router_http(
            model_str=model_str,
            call_messages=call_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=effective_reasoning_effort,
            extra_kwargs=extra_kwargs,
        )
    if _uses_direct_9router_chat_http(model_str) or _uses_direct_9router_reasoning_chat_http(
        model_str,
        provider_reasoning_effort,
    ):
        return _complete_via_9router_chat_http(
            model_str=model_str,
            call_messages=call_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=provider_reasoning_effort,
            extra_kwargs=extra_kwargs,
        )

    try:
        response: ModelResponse = litellm.completion(**call_kwargs)
    except AuthenticationError as e:
        raise LLMError(401, f"Xác thực thất bại với provider '{model_str.split('/')[0]}'. Kiểm tra API key.") from e
    except RateLimitError as e:
        raise LLMError(429, "Provider đang giới hạn tốc độ. Vui lòng thử lại sau.") from e
    except ContextWindowExceededError as e:
        raise LLMError(400, "Ngữ cảnh vượt quá giới hạn context window của model. Hãy rút ngắn nội dung.") from e
    except BadRequestError as e:
        raise LLMError(400, f"Yêu cầu không hợp lệ: {str(e)[:200]}") from e
    except ServiceUnavailableError as e:
        raise LLMError(503, f"Provider '{model_str.split('/')[0]}' tạm thời không khả dụng. Thử lại sau.") from e
    except Exception as e:
        # Map generic connection errors
        err_lower = str(e).lower()
        if "connection" in err_lower or "timeout" in err_lower:
            raise LLMError(503, f"Không thể kết nối tới provider AI. ({type(e).__name__})") from e
        raise LLMError(500, f"Lỗi LLM không xác định ({type(e).__name__}): {str(e)[:200]}") from e

    text = response.choices[0].message.content or ""
    usage = response.usage

    # Extract Anthropic cache usage stats when available
    cache_read = 0
    cache_write = 0
    try:
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        if cache_read or cache_write:
            logger.info(
                "Prompt cache: read=%d tokens write=%d tokens (model=%s)",
                cache_read, cache_write, model_str,
            )
    except Exception:
        pass

    return {
        "text": text,
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "model": model_str,
    }


def _coerce_chunk_mapping(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        data = chunk.model_dump()
        if isinstance(data, dict):
            return data
    if hasattr(chunk, "dict"):
        data = chunk.dict()
        if isinstance(data, dict):
            return data
    return {}


def _extract_text_delta_from_stream_payload(payload: dict[str, Any]) -> str:
    delta = payload.get("delta")
    if isinstance(delta, str):
        return delta

    choices = payload.get("choices") or []
    if choices:
        first_choice = choices[0] or {}
        message = first_choice.get("message") or {}
        message_content = message.get("content")
        if isinstance(message_content, str):
            return message_content

        choice_delta = first_choice.get("delta") or {}
        delta_content = choice_delta.get("content")
        if isinstance(delta_content, str):
            return delta_content
        if isinstance(delta_content, list):
            return "".join(
                str(item.get("text") or "")
                for item in delta_content
                if isinstance(item, dict)
            )

    return ""


def _extract_text_from_response_payload(payload: dict[str, Any]) -> str:

    response_data = payload.get("response") or {}
    output = response_data.get("output") or payload.get("output") or []
    collected: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content") or []
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text":
                    text_value = content_item.get("text")
                    if isinstance(text_value, str):
                        collected.append(text_value)
    return "".join(collected)


def _extract_usage_from_stream_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = (payload.get("response") or {}).get("usage")
    if not isinstance(usage, dict):
        return None, None

    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    try:
        normalized_input = int(input_tokens) if input_tokens is not None else None
    except (TypeError, ValueError):
        normalized_input = None
    try:
        normalized_output = int(output_tokens) if output_tokens is not None else None
    except (TypeError, ValueError):
        normalized_output = None
    return normalized_input, normalized_output


def _iter_9router_sse_payloads(response: requests.Response) -> Iterator[dict[str, Any]]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if isinstance(payload, dict):
            yield payload


def _stream_via_9router_http(
    *,
    model_str: str,
    call_messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    extra_kwargs: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    del reasoning_effort
    base_url = (extra_kwargs.get("api_base") or "").rstrip("/")
    api_key = (extra_kwargs.get("api_key") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("Missing 9router credentials for direct Gemini call")

    payload: dict[str, Any] = {
        "model": model_str.split("openai/", 1)[1],
        "input": call_messages,
        "stream": True,
    }
    if max_tokens:
        payload["max_output_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        f"{base_url}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
        stream=True,
    )
    if response.status_code == 401:
        raise LLMError(401, "Xác thực thất bại với provider 'openai'. Kiểm tra API key.")
    if response.status_code == 429:
        raise LLMError(429, "Provider đang giới hạn tốc độ. Vui lòng thử lại sau.")
    if response.status_code >= 500:
        raise LLMError(503, "Provider 'openai' tạm thời không khả dụng. Thử lại sau.")
    if response.status_code >= 400:
        detail = response.text.strip()[:200] or "Yêu cầu không hợp lệ"
        raise LLMError(400, f"Yêu cầu không hợp lệ: {detail}")

    collected_parts: list[str] = []
    fallback_text = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_model = model_str

    try:
        for payload_chunk in _iter_9router_sse_payloads(response):
            response_model = payload_chunk.get("model") or (payload_chunk.get("response") or {}).get("model") or response_model
            payload_input_tokens, payload_output_tokens = _extract_usage_from_stream_payload(payload_chunk)
            if payload_input_tokens is not None:
                input_tokens = payload_input_tokens
            if payload_output_tokens is not None:
                output_tokens = payload_output_tokens

            delta_text = ""
            event_type = payload_chunk.get("type")
            if event_type == "response.output_text.delta":
                delta_text = payload_chunk.get("delta") or ""
            elif event_type == "response.output_text.done" and not collected_parts:
                delta_text = payload_chunk.get("text") or ""
            elif not event_type:
                delta_text = _extract_text_delta_from_stream_payload(payload_chunk)

            if not fallback_text:
                fallback_text = _extract_text_from_response_payload(payload_chunk)

            if delta_text:
                collected_parts.append(delta_text)
                yield {"type": "delta", "text": delta_text}
    finally:
        response.close()

    full_text = "".join(collected_parts) or fallback_text
    yield {
        "type": "complete",
        "text": full_text,
        "input_tokens": input_tokens if input_tokens is not None else _estimate_token_count_from_messages(call_messages),
        "output_tokens": output_tokens if output_tokens is not None else _estimate_token_count_from_text(full_text),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model": response_model,
    }


def _stream_via_9router_chat_http(
    *,
    model_str: str,
    call_messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    extra_kwargs: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    base_url = (extra_kwargs.get("api_base") or "").rstrip("/")
    api_key = (extra_kwargs.get("api_key") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("Missing 9router credentials for direct chat call")

    payload: dict[str, Any] = {
        "model": model_str.split("openai/", 1)[1],
        "messages": call_messages,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
        stream=True,
    )
    if response.status_code == 401:
        raise LLMError(401, "Xác thực thất bại với provider 'openai'. Kiểm tra API key.")
    if response.status_code == 429:
        raise LLMError(429, "Provider đang giới hạn tốc độ. Vui lòng thử lại sau.")
    if response.status_code >= 500:
        raise LLMError(503, "Provider 'openai' tạm thời không khả dụng. Thử lại sau.")
    if response.status_code >= 400:
        detail = response.text.strip()[:200] or "Yêu cầu không hợp lệ"
        raise LLMError(400, f"Yêu cầu không hợp lệ: {detail}")

    collected_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_model = model_str

    try:
        for payload_chunk in _iter_9router_sse_payloads(response):
            response_model = payload_chunk.get("model") or response_model
            payload_input_tokens, payload_output_tokens = _extract_usage_from_stream_payload(payload_chunk)
            if payload_input_tokens is not None:
                input_tokens = payload_input_tokens
            if payload_output_tokens is not None:
                output_tokens = payload_output_tokens

            delta_text = _extract_text_delta_from_stream_payload(payload_chunk)
            if delta_text:
                collected_parts.append(delta_text)
                yield {"type": "delta", "text": delta_text}
    finally:
        response.close()

    full_text = "".join(collected_parts)
    yield {
        "type": "complete",
        "text": full_text,
        "input_tokens": input_tokens if input_tokens is not None else _estimate_token_count_from_messages(call_messages),
        "output_tokens": output_tokens if output_tokens is not None else _estimate_token_count_from_text(full_text),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model": response_model,
    }


def stream_complete(
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    model: str | None = None,
    images: list[str | bytes] | None = None,
    image_media_types: list[str] | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> Iterator[dict[str, Any]]:
    model_str = resolve_model(model) if not images else resolve_vision_model(model)
    extra_kwargs = _provider_kwargs(model_str)

    processed_images: list[str | bytes] = []
    processed_media_types: list[str] = []
    ocr_texts: list[str] = []

    for idx, img in enumerate(images or []):
        mt = (image_media_types or [])[idx] if idx < len(image_media_types or []) else "image/jpeg"
        p_img, p_mt, ocr_text = _preprocess_image(img, mt)
        if ocr_text:
            ocr_texts.append(ocr_text)
            logger.info("Image %d: OCR extracted %d chars, skipping vision", idx + 1, len(ocr_text))
        else:
            processed_images.append(p_img)
            processed_media_types.append(p_mt)

    call_messages = list(messages)
    effective_system = system
    if ocr_texts:
        ocr_blocks = "\n\n".join(
            f"--- Ảnh {i+1} ---\n{t}"
            for i, t in enumerate(ocr_texts)
        )
        ocr_system_note = (
            "Hệ thống đã tự động trích xuất văn bản từ ảnh đính kèm bằng OCR. "
            "Hãy sử dụng nội dung dưới đây để trả lời câu hỏi của người dùng một cách tự nhiên, "
            "như thể bạn đã phân tích ảnh trực tiếp. "
            "Không đề cập đến việc OCR hay không thể xem ảnh.\n\n"
            f"Nội dung trích xuất từ ảnh:\n{ocr_blocks}"
        )
        effective_system = (effective_system + "\n\n" + ocr_system_note) if effective_system else ocr_system_note

    if processed_images:
        call_messages = _inject_images(call_messages, processed_images, processed_media_types)

    call_messages, system_after_cache = _apply_prompt_caching(call_messages, effective_system, model_str)
    if system_after_cache:
        call_messages = [{"role": "system", "content": system_after_cache}] + call_messages

    call_kwargs: dict[str, Any] = {
        "model": model_str,
        "messages": call_messages,
        "max_tokens": max_tokens,
        "stream": True,
        **extra_kwargs,
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature

    effective_reasoning_effort = _normalize_reasoning_effort(reasoning_effort) or _normalize_reasoning_effort(settings.llm_reasoning_effort)
    provider_reasoning_effort = _provider_reasoning_effort(model_str, effective_reasoning_effort)
    if provider_reasoning_effort and _supports_reasoning_effort(model_str):
        call_kwargs["reasoning_effort"] = provider_reasoning_effort
        if model_str in {"openai/gh/claude-sonnet-4.6", "openai/gh/claude-opus-4.6"} | KILOCODE_REASONING_MODELS:
            call_kwargs["allowed_openai_params"] = ["reasoning_effort"]
    elif effective_reasoning_effort:
        logger.info(
            "Ignoring reasoning_effort=%s for unsupported model=%s",
            effective_reasoning_effort,
            model_str,
        )

    logger.debug(
        "LiteLLM stream call model=%s messages=%d vision_imgs=%d ocr_imgs=%d max_tokens=%d",
        model_str, len(call_messages), len(processed_images), len(ocr_texts), max_tokens,
    )

    if _uses_direct_9router_http(model_str):
        yield from _stream_via_9router_http(
            model_str=model_str,
            call_messages=call_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=effective_reasoning_effort,
            extra_kwargs=extra_kwargs,
        )
        return

    if _uses_direct_9router_chat_http(model_str) or _uses_direct_9router_reasoning_chat_http(
        model_str,
        provider_reasoning_effort,
    ):
        yield from _stream_via_9router_chat_http(
            model_str=model_str,
            call_messages=call_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=provider_reasoning_effort,
            extra_kwargs=extra_kwargs,
        )
        return

    collected_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens = 0
    cache_write_tokens = 0
    response_model = model_str

    try:
        response_stream = litellm.completion(**call_kwargs)
        for chunk in response_stream:
            payload = _coerce_chunk_mapping(chunk)
            if payload:
                response_model = payload.get("model") or response_model
                payload_input_tokens, payload_output_tokens = _extract_usage_from_stream_payload(payload)
                if payload_input_tokens is not None:
                    input_tokens = payload_input_tokens
                if payload_output_tokens is not None:
                    output_tokens = payload_output_tokens
                usage = payload.get("usage") or {}
                if isinstance(usage, dict):
                    cache_read_tokens = int(usage.get("cache_read_input_tokens", cache_read_tokens) or cache_read_tokens)
                    cache_write_tokens = int(usage.get("cache_creation_input_tokens", cache_write_tokens) or cache_write_tokens)

            delta_text = _extract_text_delta_from_stream_payload(payload)
            if delta_text:
                collected_parts.append(delta_text)
                yield {"type": "delta", "text": delta_text}
    except AuthenticationError as e:
        raise LLMError(401, f"Xác thực thất bại với provider '{model_str.split('/')[0]}'. Kiểm tra API key.") from e
    except RateLimitError as e:
        raise LLMError(429, "Provider đang giới hạn tốc độ. Vui lòng thử lại sau.") from e
    except ContextWindowExceededError as e:
        raise LLMError(400, "Ngữ cảnh vượt quá giới hạn context window của model. Hãy rút ngắn nội dung.") from e
    except BadRequestError as e:
        raise LLMError(400, f"Yêu cầu không hợp lệ: {str(e)[:200]}") from e
    except ServiceUnavailableError as e:
        raise LLMError(503, f"Provider '{model_str.split('/')[0]}' tạm thời không khả dụng. Thử lại sau.") from e
    except Exception as e:
        err_lower = str(e).lower()
        if "connection" in err_lower or "timeout" in err_lower:
            raise LLMError(503, f"Không thể kết nối tới provider AI. ({type(e).__name__})") from e
        raise LLMError(500, f"Lỗi LLM không xác định ({type(e).__name__}): {str(e)[:200]}") from e

    full_text = "".join(collected_parts)
    yield {
        "type": "complete",
        "text": full_text,
        "input_tokens": input_tokens if input_tokens is not None else _estimate_token_count_from_messages(call_messages),
        "output_tokens": output_tokens if output_tokens is not None else _estimate_token_count_from_text(full_text),
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "model": response_model,
    }


def _append_text_to_last_user(messages: list[dict], extra_text: str) -> list[dict]:
    """Append text to the last user message (or add a new user message)."""
    result = [dict(m) for m in messages]
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            content = result[i]["content"]
            if isinstance(content, str):
                result[i] = {**result[i], "content": content + "\n\n" + extra_text}
            elif isinstance(content, list):
                result[i] = {**result[i], "content": content + [{"type": "text", "text": extra_text}]}
            return result
    result.append({"role": "user", "content": extra_text})
    return result


def _inject_images(
    messages: list[dict],
    images: list[str | bytes],
    media_types: list[str],
) -> list[dict]:
    """Inject image blocks into the last user message (or append new one)."""
    if not images:
        return list(messages)

    result = [dict(m) for m in messages]
    last_user_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            last_user_idx = i
            break

    image_blocks = [
        _make_image_block(img, media_types[idx] if idx < len(media_types) else "image/jpeg")
        for idx, img in enumerate(images)
    ]

    if last_user_idx is not None:
        existing_content = result[last_user_idx]["content"]
        if isinstance(existing_content, str):
            new_content: list[dict] = [{"type": "text", "text": existing_content}]
        elif isinstance(existing_content, list):
            new_content = list(existing_content)
        else:
            new_content = []
        new_content.extend(image_blocks)
        result[last_user_idx] = {**result[last_user_idx], "content": new_content}
    else:
        result.append({"role": "user", "content": image_blocks})

    return result


# ── Image captioning for document ingestion ───────────────────────────────────

def caption_image(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    context_hint: str = "",
    model: str | None = None,
) -> str:
    """Generate a text description of an image for RAG ingestion.

    Pipeline:
    1. Preprocess image (resize + format conversion).
    2. If OCR detection finds text-heavy image → return OCR text directly
       (no vision API call, zero extra cost).
    3. Otherwise call vision model and return its description.

    Returns empty string when captioning is disabled or fails.
    """
    if not settings.llm_image_captioning_enabled:
        return ""
    if not image_bytes:
        return ""

    # ── Step 1: preprocess ────────────────────────────────────────────────
    _, final_mt, ocr_text = _preprocess_image(image_bytes, media_type)

    # ── Step 2: OCR path (free) ───────────────────────────────────────────
    if ocr_text and len(ocr_text.split()) >= 10:
        logger.debug("caption_image: using OCR text (%d words)", len(ocr_text.split()))
        return ocr_text

    # ── Step 3: Vision path ───────────────────────────────────────────────
    prompt = (
        "Mô tả nội dung hình ảnh này một cách chi tiết bằng tiếng Việt, "
        "tập trung vào văn bản, số liệu, biểu đồ, sơ đồ hoặc thông tin có thể tìm kiếm được. "
        "Không thêm lời giải thích hay nhận xét ngoài nội dung quan sát được."
    )
    if context_hint:
        prompt += f"\nNgữ cảnh tài liệu: {context_hint}"

    try:
        # Re-preprocess to get final bytes (already done above but _preprocess_image
        # returns the image only when use_vision=True path taken)
        from app.services.image_processor import preprocess_for_llm
        prep = preprocess_for_llm(
            image_bytes, media_type,
            max_dimension=settings.llm_image_max_dimension,
            ocr_enabled=False,  # already handled above
        )
        final_bytes = prep.image_bytes
        final_mt = prep.media_type

        result = complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.llm_image_caption_max_tokens,
            model=resolve_vision_model(model),
            images=[final_bytes],
            image_media_types=[final_mt],
        )
        return result["text"].strip()
    except LLMError as e:
        logger.warning("Image captioning failed (non-fatal): %s", e.detail)
        return ""
    except Exception as e:
        logger.warning("Image captioning unexpected error (non-fatal): %s", e)
        return ""

