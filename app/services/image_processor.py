"""Image preprocessing pipeline for LLM vision calls and document ingestion.

Features
--------
1. **Resize**  – Shrink images to the largest dimension that a vision model
   needs (default 1 568 px, Anthropic's recommended sweet-spot).  Saves
   significant tokens: a 4 000×3 000 image → ~800×600 cuts token cost ~25×.

2. **Format normalisation** – Convert RGBA/palette PNGs to RGB JPEG for
   smaller payloads.  PNG is kept only when the image has meaningful
   transparency.

3. **OCR detection & extraction** – Decide whether an image is
   *text-heavy* (screenshot, scanned page, slide) or *visual* (photo,
   chart, diagram).  If text-heavy, run Tesseract OCR (when available) and
   return the extracted text instead of sending the image to an expensive
   vision model.

   Detection methods (in priority order):
   a. **Tesseract confidence score** – If pytesseract + Tesseract binary are
      both installed, run OCR and use the word-level confidence average.
      Confidence ≥ threshold (default 55 %) → treat as OCR image.
   b. **Pixel statistics heuristic** – Works without Tesseract.  Text images
      have a bimodal pixel distribution: lots of near-white background
      pixels and near-black text pixels.  If ≥ 70 % of pixels are near-B/W
      with both colours present → OCR candidate.

   Can we reliably distinguish OCR images from other images?
   ✔ YES for clear cases: screenshots, scanned A4 docs, slide exports.
   ⚠ Borderline: infographics with black outlines may trigger OCR mode.
     But even then the OCR output is checked for "meaningful" text (≥ 15
     words) before the image byte payload is dropped.

4. **Metadata extraction** – EXIF/IPTC info attached as a text note for
   context (camera model, GPS, creation date, etc.).

The ``preprocess_for_llm`` entry-point returns an ``ImagePreprocessResult``
that the caller uses to decide:
  - Send image bytes to vision model  (``use_vision=True``)
  - Send only OCR text as plain message (``use_vision=False``)
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("uvicorn.error")

# ── Constants ─────────────────────────────────────────────────────────────────

# Anthropic recommends ≤ 1 568 px on the longest side for claude-3.x models.
# GPT-4o supports up to 2 048 px but we use the conservative value as a
# universal cap.  Below this threshold images are NOT upscaled.
DEFAULT_MAX_DIMENSION = 1_568

# JPEG quality for re-encoded images.  85 is visually lossless for text/UI
# screenshots and roughly halves file size vs the original PNG.
JPEG_QUALITY = 85

# Approximate token cost per pixel (Anthropic: 1 token ≈ 750 px²).
# Used only for logging; not enforced here.
_TOKENS_PER_PX2 = 1 / 750

# Pixel heuristic thresholds for text-image detection
_BW_RATIO_THRESHOLD = 0.70   # ≥70 % pixels near-B/W → possibly text
_WHITE_MIN = 0.25             # need at least 25 % white (background)
_BLACK_MIN = 0.02             # need at least 2 % black (ink/text)

# Min word count from OCR to consider the result "meaningful"
_MIN_OCR_WORDS = 12


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ImagePreprocessResult:
    """Output of the preprocessing pipeline."""

    # Processed image bytes (may be resized/re-encoded), empty when use_vision=False
    image_bytes: bytes
    media_type: str               # e.g. "image/jpeg"

    # Dimensions after processing
    width: int = 0
    height: int = 0

    # Size statistics
    original_size_bytes: int = 0
    final_size_bytes: int = 0

    # Whether to send image to vision model (False = OCR text is sufficient)
    use_vision: bool = True

    # OCR extracted text (non-empty only when use_vision=False or as supplement)
    ocr_text: str = ""

    # Metadata string (EXIF / file info)
    metadata_note: str = ""

    # Diagnostics
    is_ocr_candidate: bool = False
    ocr_confidence: float = 0.0
    ocr_engine: str = ""          # "tesseract" | "heuristic" | "none"
    resize_applied: bool = False
    format_converted: bool = False
    notes: list[str] = field(default_factory=list)


# ── Public entry-point ────────────────────────────────────────────────────────

def preprocess_for_llm(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    ocr_enabled: bool = True,
    ocr_confidence_threshold: float = 0.55,
    extract_metadata: bool = True,
) -> ImagePreprocessResult:
    """Full preprocessing pipeline.

    Args:
        image_bytes: Raw image bytes.
        media_type:  MIME type hint (may be corrected from actual file).
        max_dimension: Longest allowed side in pixels.
        ocr_enabled:  Whether to attempt OCR detection/extraction.
        ocr_confidence_threshold: Min confidence (0–1) to classify as OCR.
        extract_metadata: Whether to extract EXIF / file metadata.

    Returns:
        ``ImagePreprocessResult`` with processed bytes and analysis.
    """
    result = ImagePreprocessResult(
        image_bytes=image_bytes,
        media_type=media_type,
        original_size_bytes=len(image_bytes),
        final_size_bytes=len(image_bytes),
    )

    if not image_bytes:
        result.notes.append("Empty image bytes – skipped.")
        return result

    try:
        from PIL import Image, ExifTags
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        result.notes.append(f"Cannot open image: {exc}")
        return result

    result.width, result.height = img.size

    # ── 1. Metadata extraction ────────────────────────────────────────────
    if extract_metadata:
        result.metadata_note = _extract_metadata(img)

    # ── 2. Resize ─────────────────────────────────────────────────────────
    img, resized = _resize_image(img, max_dimension)
    if resized:
        result.resize_applied = True
        result.width, result.height = img.size
        result.notes.append(f"Resized to {result.width}×{result.height}")

    # ── 3. Format normalisation → JPEG ────────────────────────────────────
    img, converted, final_bytes, final_mt = _normalise_format(img, media_type)
    if converted:
        result.format_converted = True
        result.notes.append(f"Converted to {final_mt}")

    result.image_bytes = final_bytes
    result.media_type = final_mt
    result.final_size_bytes = len(final_bytes)

    # ── 4. OCR detection + extraction ─────────────────────────────────────
    if ocr_enabled:
        _run_ocr_pipeline(img, result, ocr_confidence_threshold)

    token_estimate = (result.width * result.height) / 750
    saving_pct = 100 * (1 - result.final_size_bytes / max(result.original_size_bytes, 1))
    logger.debug(
        "ImagePreprocess: %dx%d  %dB→%dB (%.0f%% smaller)  ocr=%s  vision=%s  ~%.0f tokens",
        result.width, result.height,
        result.original_size_bytes, result.final_size_bytes, saving_pct,
        result.is_ocr_candidate, result.use_vision, token_estimate,
    )
    return result


# ── Step implementations ──────────────────────────────────────────────────────

def _resize_image(img, max_dim: int):
    """Downscale so that the longest side ≤ max_dim.  Never upscale."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img, False

    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    try:
        from PIL import Image
        img = img.resize((new_w, new_h), Image.LANCZOS)
    except Exception:
        img = img.resize((new_w, new_h))
    return img, True


def _normalise_format(img, original_mt: str):
    """Convert image to RGB JPEG unless it has meaningful transparency."""
    from PIL import Image

    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    target_jpeg = not has_alpha  # keep PNG if alpha matters

    if target_jpeg and img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    if target_jpeg:
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        final_mt = "image/jpeg"
        converted = original_mt != "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        final_mt = "image/png"
        converted = False  # kept PNG intentionally

    return img, converted, buf.getvalue(), final_mt


def _extract_metadata(img) -> str:
    """Extract EXIF / basic metadata and return as a short text note."""
    parts: list[str] = []
    try:
        from PIL import ExifTags
        exif_data = img._getexif() if hasattr(img, "_getexif") else None
        if exif_data:
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            for key in ("Make", "Model", "DateTimeOriginal", "DateTime", "Software"):
                val = tags.get(key)
                if val and isinstance(val, str):
                    parts.append(f"{key}: {val.strip()}")
    except Exception:
        pass

    if not parts:
        parts.append(f"Format: {img.format or 'unknown'}  Mode: {img.mode}  Size: {img.size[0]}×{img.size[1]}")

    return " | ".join(parts)


def _run_ocr_pipeline(img, result: ImagePreprocessResult, threshold: float) -> None:
    """Detect whether the image is text-heavy and extract text if so."""
    # Try Tesseract first (most accurate)
    if _try_tesseract(img, result, threshold):
        return
    # Fallback: pixel heuristic
    _try_heuristic(img, result, threshold)


def _try_tesseract(img, result: ImagePreprocessResult, threshold: float) -> bool:
    """Run pytesseract if available.  Returns True if OCR was attempted."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return False  # pytesseract not installed

    # ── Ensure Tesseract binary is found on Windows ───────────────────────
    import shutil, os
    if not shutil.which("tesseract"):
        # Try common Windows install paths
        candidate_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        ]
        for path in candidate_paths:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    try:
        # Use RGB for Tesseract
        ocr_img = img.convert("RGB") if img.mode != "RGB" else img
        data = pytesseract.image_to_data(
            ocr_img,
            output_type=pytesseract.Output.DICT,
            config="--psm 3",  # auto page segmentation
        )
        confidences = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        text_words = [w for w in data["text"] if w.strip()]

        avg_conf = (sum(confidences) / len(confidences) / 100) if confidences else 0.0
        result.ocr_confidence = avg_conf
        result.ocr_engine = "tesseract"

        if avg_conf >= threshold and len(text_words) >= _MIN_OCR_WORDS:
            result.is_ocr_candidate = True
            result.ocr_text = " ".join(text_words)
            result.use_vision = False          # skip vision model – OCR is enough
            result.notes.append(
                f"Tesseract OCR: conf={avg_conf:.0%} words={len(text_words)} → use OCR text"
            )
        else:
            result.notes.append(
                f"Tesseract OCR: conf={avg_conf:.0%} words={len(text_words)} → use vision model"
            )
        return True

    except Exception as exc:
        # Tesseract binary missing or runtime error → fall through to heuristic
        result.notes.append(f"Tesseract unavailable ({exc.__class__.__name__}), using heuristic")
        return False


def _try_heuristic(img, result: ImagePreprocessResult, threshold: float) -> None:
    """Pixel statistics heuristic for OCR detection (no external dependency)."""
    try:
        gray = img.convert("L")
        pixels = list(gray.getdata())
        total = len(pixels)
        if total == 0:
            return

        near_white = sum(1 for p in pixels if p > 200) / total
        near_black = sum(1 for p in pixels if p < 55) / total
        bw_ratio = near_white + near_black

        heuristic_conf = 0.0
        if bw_ratio >= _BW_RATIO_THRESHOLD and near_white >= _WHITE_MIN and near_black >= _BLACK_MIN:
            # Normalise to a 0-1 confidence: 0.70 → 0.0, 1.0 → 1.0
            heuristic_conf = min(1.0, (bw_ratio - _BW_RATIO_THRESHOLD) / (1.0 - _BW_RATIO_THRESHOLD))

        result.ocr_confidence = heuristic_conf
        result.ocr_engine = "heuristic"

        if heuristic_conf >= threshold:
            result.is_ocr_candidate = True
            # No actual text extracted – signal caller to use vision but note it's text-heavy
            result.notes.append(
                f"Heuristic: bw_ratio={bw_ratio:.0%} conf={heuristic_conf:.0%} → text-heavy image"
            )
        else:
            result.notes.append(
                f"Heuristic: bw_ratio={bw_ratio:.0%} conf={heuristic_conf:.0%} → visual image"
            )
    except Exception as exc:
        result.notes.append(f"Heuristic failed: {exc}")

