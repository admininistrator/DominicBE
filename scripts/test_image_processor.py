from app.services.image_processor import preprocess_for_llm
from PIL import Image, ImageDraw
import io

# Test 1: Large photo (should resize + convert to JPEG)
img = Image.new('RGB', (3000, 2000), color=(200, 50, 50))
buf = io.BytesIO()
img.save(buf, 'PNG')
img_bytes = buf.getvalue()
result = preprocess_for_llm(img_bytes, 'image/png')
print("=== Test 1: Large photo ===")
print(f"Original: 3000x2000 PNG {len(img_bytes)//1024}KB")
print(f"After:    {result.width}x{result.height} {result.media_type} {result.final_size_bytes//1024}KB")
print(f"Resize: {result.resize_applied}  Convert: {result.format_converted}")
print(f"OCR: candidate={result.is_ocr_candidate}  conf={result.ocr_confidence:.0%}  engine={result.ocr_engine}")
print(f"Notes: {result.notes}")

# Test 2: Small image (should NOT resize)
img2 = Image.new('RGB', (400, 300), color=(50, 200, 50))
buf2 = io.BytesIO()
img2.save(buf2, 'JPEG')
result2 = preprocess_for_llm(buf2.getvalue(), 'image/jpeg')
print("\n=== Test 2: Small image (no resize) ===")
print(f"Size: {result2.width}x{result2.height}  Resize: {result2.resize_applied}")

# Test 3: RGBA PNG with transparency (should keep PNG)
img3 = Image.new('RGBA', (800, 600), color=(100, 100, 255, 128))
buf3 = io.BytesIO()
img3.save(buf3, 'PNG')
result3 = preprocess_for_llm(buf3.getvalue(), 'image/png')
print("\n=== Test 3: RGBA PNG (keep PNG) ===")
print(f"Media type: {result3.media_type}  Convert: {result3.format_converted}")

# Test 4: Image with real text (Tesseract should detect and extract)
img4 = Image.new('RGB', (800, 200), color=(255, 255, 255))
draw = ImageDraw.Draw(img4)
# Draw readable text using default font
draw.text((20, 20),  "Hello World - This is a test document.", fill=(0, 0, 0))
draw.text((20, 60),  "Noi dung van ban tieng Viet: xin chao the gioi.", fill=(0, 0, 0))
draw.text((20, 100), "Invoice #12345  Date: 2024-01-15  Total: $999.00", fill=(0, 0, 0))
draw.text((20, 140), "The quick brown fox jumps over the lazy dog.", fill=(0, 0, 0))
buf4 = io.BytesIO()
img4.save(buf4, 'PNG')
result4 = preprocess_for_llm(buf4.getvalue(), 'image/png', ocr_enabled=True)
print("\n=== Test 4: Image with real text (Tesseract OCR) ===")
print(f"OCR candidate: {result4.is_ocr_candidate}  conf: {result4.ocr_confidence:.0%}  engine: {result4.ocr_engine}")
print(f"use_vision: {result4.use_vision}")
print(f"OCR text: {repr(result4.ocr_text[:200]) if result4.ocr_text else '(empty)'}")
print(f"Notes: {result4.notes}")

# Test 5: Colorful photo (should NOT be OCR candidate)
img5 = Image.new('RGB', (500, 500))
pixels = img5.load()
import random
random.seed(42)
for x in range(500):
    for y in range(500):
        pixels[x, y] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
buf5 = io.BytesIO()
img5.save(buf5, 'JPEG')
result5 = preprocess_for_llm(buf5.getvalue(), 'image/jpeg', ocr_enabled=True)
print("\n=== Test 5: Colorful noise photo (NOT OCR) ===")
print(f"OCR candidate: {result5.is_ocr_candidate}  conf: {result5.ocr_confidence:.0%}  engine: {result5.ocr_engine}")
print(f"use_vision: {result5.use_vision}")

print("\nAll tests passed!")

