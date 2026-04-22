"""Unified LLM provider layer using LiteLLM.

Provides a single interface to call any model (Anthropic, OpenAI, Gemini, Azure, etc.)
by changing only the model string in config.  No other code needs to change when
switching providers.

Model string examples:
    "anthropic/claude-3-5-haiku-latest"   ← Anthropic
    "openai/gpt-4o"                       ← OpenAI
    "gemini/gemini-1.5-pro"               ← Google Gemini
    "azure/gpt-4o"                        ← Azure OpenAI
    "ollama/llama3"                       ← local Ollama

Image processing pipeline (applied automatically):
    1. Resize  – longest side ≤ LLM_IMAGE_MAX_DIMENSION (default 1 568 px)
    2. Format  – RGBA/palette PNG → RGB JPEG (smaller payload)
    3. OCR     – text-heavy images (screenshots, scanned docs) are identified
                 via Tesseract confidence or pixel-statistics heuristic.
                 When OCR text is extracted, the image bytes are *not* sent to
                 the vision model → significant token savings.

Prompt caching (Anthropic only, transparent no-op for other providers):
    When LLM_PROMPT_CACHING_ENABLED=true and the system prompt is long
    enough (≥ LLM_PROMPT_CACHING_MIN_CHARS), ``cache_control`` blocks are
    injected so Anthropic can cache the prefix.
    Cache read cost: 10 % of normal.  Cache write: 125 % (amortised fast).
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from typing import Any

import litellm
from litellm import ModelResponse
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    RateLimitError,
    ServiceUnavailableError,
)

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")

# ── Silence verbose litellm logging ──────────────────────────────────────────
litellm.suppress_debug_info = True
litellm.set_verbose = False


# ── Model resolution ──────────────────────────────────────────────────────────

def resolve_model(model: str | None = None) -> str:
    """Return the full LiteLLM model string to use.

    Priority:
    1. Explicit ``model`` argument
    2. ``LLM_MODEL`` env / settings (e.g. "anthropic/claude-3-5-haiku-latest")
    3. Legacy ``ANTHROPIC_MODEL`` env prefixed with "anthropic/"
    4. Hardcoded safe default
    """
    if model and model.strip():
        return model.strip()

    configured = (settings.llm_model or "").strip()
    if configured:
        return configured

    # Backward-compat: use anthropic_model with provider prefix
    legacy = (settings.anthropic_model or "").strip()
    if legacy:
        if "/" not in legacy:
            return f"anthropic/{legacy}"
        return legacy

    return "anthropic/claude-3-5-haiku-latest"


def resolve_vision_model(model: str | None = None) -> str:
    """Return vision-capable model string."""
    if model and model.strip():
        return model.strip()
    configured = (settings.llm_vision_model or "").strip()
    if configured:
        return configured
    # Fall back to main model – most modern models are vision-capable
    return resolve_model()


# ── API key / extra kwargs per provider ──────────────────────────────────────

def _provider_name(model_str: str) -> str:
    """Extract provider name from a LiteLLM model string (e.g. 'anthropic/...' → 'anthropic')."""
    return model_str.split("/")[0].lower() if "/" in model_str else "anthropic"


def _provider_kwargs(model_str: str) -> dict[str, Any]:
    """Inject provider-specific API keys / base URLs into litellm call kwargs."""
    kwargs: dict[str, Any] = {}
    provider = model_str.split("/")[0].lower() if "/" in model_str else "anthropic"

    if provider == "anthropic":
        api_key = (settings.anthropic_api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Configure it in .env or environment variables."
            )
        kwargs["api_key"] = api_key
        base_url = (settings.anthropic_base_url or "").strip()
        if base_url:
            kwargs["api_base"] = base_url

    elif provider in ("openai", "azure"):
        api_key = (settings.openai_api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Configure it in .env or environment variables."
            )
        kwargs["api_key"] = api_key
        base_url = (settings.openai_base_url or "").strip()
        if base_url:
            kwargs["api_base"] = base_url

    elif provider == "gemini":
        api_key = (settings.gemini_api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Configure it in .env or environment variables."
            )
        kwargs["api_key"] = api_key

    # For other providers (ollama, cohere, etc.) litellm handles auth from env.
    return kwargs


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


# ── Prompt caching (Anthropic only) ──────────────────────────────────────────

def _apply_prompt_caching(
    call_messages: list[dict],
    system: str | None,
    model_str: str,
) -> tuple[list[dict], str | None]:
    """Inject Anthropic ``cache_control`` blocks where beneficial.

    Rules:
    - Only applied when provider == "anthropic" and caching is enabled.
    - System prompt: cached when len(system) ≥ LLM_PROMPT_CACHING_MIN_CHARS.
    - Last two human turns: mark with cache_control so repeated queries
      against the same long context get cache hits on the knowledge block.

    Returns:
        (modified_messages, modified_system)
        modified_system is None when it should be passed as a system message
        inside the messages list (Anthropic SDK style).
    """
    if not settings.llm_prompt_caching_enabled:
        return call_messages, system
    if _provider_name(model_str) != "anthropic":
        return call_messages, system

    min_chars = settings.llm_prompt_caching_min_chars
    cached_system = system

    # Cache system prompt when large enough
    if system and len(system) >= min_chars:
        # LiteLLM forwards cache_control when system is passed as a list of blocks
        # We signal this by attaching metadata to the system string – but since
        # LiteLLM doesn't support that directly, we instead prepend a system
        # message with cache_control content array.
        cached_system = None  # will be injected into messages below
        system_block = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        call_messages = [system_block] + call_messages
        logger.debug("Prompt caching: system prompt cached (%d chars)", len(system))

    # Cache the last user message that has substantial context (e.g. RAG block).
    # We find user messages with long content and add cache_control to their
    # last text block.
    cacheable_indices = [
        i for i, m in enumerate(call_messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and len(m["content"]) >= min_chars
    ]
    # Cache at most the 2 most recent long user messages
    for idx in cacheable_indices[-2:]:
        msg = call_messages[idx]
        content = msg["content"]
        if isinstance(content, str):
            call_messages[idx] = {
                **msg,
                "content": [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ],
            }
            logger.debug("Prompt caching: user message idx=%d cached (%d chars)", idx, len(content))

    return call_messages, cached_system


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

    logger.debug(
        "LiteLLM call model=%s messages=%d vision_imgs=%d ocr_imgs=%d max_tokens=%d",
        model_str, len(call_messages), len(processed_images), len(ocr_texts), max_tokens,
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

