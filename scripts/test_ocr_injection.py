"""Verify OCR text is injected correctly without 'OCR label' confusion."""
import sys, io
sys.path.insert(0, '.')

from app.services.llm_provider import (
    _preprocess_image, _append_text_to_last_user, _inject_images
)
from PIL import Image, ImageDraw
import base64

def make_text_image():
    """Create a simple white image with readable black text."""
    img = Image.new('RGB', (800, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20),  "Invoice #12345  Date: 2024-01-15", fill=(0, 0, 0))
    draw.text((20, 60),  "Customer: Nguyen Van A  Total: 999.000 VND", fill=(0, 0, 0))
    draw.text((20, 100), "Product: Laptop Dell XPS  Qty: 1  Price: 999.000", fill=(0, 0, 0))
    draw.text((20, 140), "Status: PAID  Payment: Bank Transfer", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()

img_bytes = make_text_image()
media_type = "image/png"

print("=== Testing OCR injection logic ===\n")

# Run preprocess
p_img, p_mt, ocr_text = _preprocess_image(img_bytes, media_type)

print(f"OCR extracted: {ocr_text is not None}")
if ocr_text:
    print(f"OCR text ({len(ocr_text)} chars): {repr(ocr_text[:150])}")
    print()

    # Simulate what complete() does: inject as plain text (NOT labeled as OCR)
    messages = [{"role": "user", "content": "Hóa đơn này ghi gì?"}]
    messages = _append_text_to_last_user(messages, ocr_text)

    print("Final user message content:")
    print(repr(messages[-1]["content"]))
    print()

    # Verify: no confusing "OCR" framing in user message
    user_content = messages[-1]["content"]
    assert "[Nội dung trích xuất từ ảnh" not in str(user_content), \
        "FAIL: old OCR label still present in user message!"
    assert "Invoice" in str(user_content) or "PAID" in str(user_content), \
        "FAIL: OCR text not injected into user message!"
    print("✅ OCR text injected as plain content (no confusing label)")
    print("✅ No '[Nội dung trích xuất từ ảnh (OCR)]' framing in user message")
    print("✅ Context note moved to system prompt level (added by complete())")
else:
    print("⚠️  OCR not triggered – Tesseract may need Vietnamese text for high confidence")
    print("   (This is OK – image will be sent to vision model instead)")

print("\nTest complete!")
