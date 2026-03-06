# ocr_processor.py
import os
import tempfile
import cv2
import pytesseract
import re
from pdf2image import convert_from_path
from PIL import Image
import numpy as np


# ==========================================
# 🔹 SIMPLE IMAGE OCR
# ==========================================
def ocr_image_simple(image_path: str) -> str:
    try:
        print(f"Processing image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=30)

        # Increase contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Adaptive thresholding (handles uneven lighting)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

        # Scale up small images (helps Tesseract a lot)
        h, w = binary.shape
        if w < 1800:
            scale = 1800 / w
            binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Tesseract config: treat as a full page, OEM 3 = best LSTM mode
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(binary, lang="eng", config=config)

        print(f"Extracted {len(text)} characters")
        return text.strip()

    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


# ==========================================
# 🔹 SIMPLE PDF OCR
# ==========================================
def ocr_pdf_simple(pdf_path: str) -> str:
    try:
        print(f"Processing PDF: {pdf_path}")

        pages = convert_from_path(pdf_path, dpi=300)

        all_text = []

        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)}")

            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"temp_{os.urandom(4).hex()}.png"
            )

            page.save(temp_path, "PNG")

            page_text = ocr_image_simple(temp_path)

            if page_text.strip():
                all_text.append(page_text)

            try:
                os.remove(temp_path)
            except:
                pass

        combined = "\n\n".join(all_text)
        print(f"Total extracted characters: {len(combined)}")

        return combined

    except Exception as e:
        print(f"PDF OCR Error: {e}")
        return ""


# ==========================================
# 🔹 BOOK STRUCTURE DETECTION
# ==========================================
def structure_book_text(full_text: str) -> dict:
    if not full_text.strip():
        return {"book_title": "Unknown Title", "chapters": []}

    # Merge broken OCR lines
    raw_lines = full_text.split("\n")
    merged_lines = []
    buffer = ""
    for line in raw_lines:
        line = line.strip()
        if not line:
            if buffer:
                merged_lines.append(buffer.strip())
                buffer = ""
            continue
        if buffer and not buffer.endswith((".", "!", "?", ":", ";")):
            buffer += " " + line
        else:
            if buffer:
                merged_lines.append(buffer.strip())
            buffer = line
    if buffer:
        merged_lines.append(buffer.strip())

    book_title = merged_lines[0] if merged_lines else "Unknown Title"

    # Detect chapters
    chapters = []
    current_chapter = None
    for line in merged_lines:
        words = line.split()
        heading_score = 0
        if 2 <= len(words) <= 8:
            heading_score += 1
        upper_ratio = sum(c.isupper() for c in line) / max(len(line), 1)
        if upper_ratio > 0.6:
            heading_score += 2
        if line.istitle():
            heading_score += 1
        if re.search(r'\bchapter\b', line, re.IGNORECASE):
            heading_score += 3

        if heading_score >= 3:
            # Always append current chapter if exists, no length check
            if current_chapter:
                chapters.append(current_chapter)
            current_chapter = {"heading": line, "content": []}
        else:
            if current_chapter:
                current_chapter["content"].append(line)

    if current_chapter:
        chapters.append(current_chapter)  # always append last chapter

    # Format structured chapters
    structured_chapters = []
    for idx, ch in enumerate(chapters):
        structured_chapters.append({
            "article_id": f"chapter_{idx+1}",
            "heading": ch["heading"],
            "subheading": "",
            "body": ch["content"],
            "full_text": "\n".join(ch["content"]).strip()
        })

    print(f"Detected {len(structured_chapters)} structured chapters")
    return {"book_title": book_title, "chapters": structured_chapters}


# ==========================================
# 🔹 MAIN BOOK TEXT EXTRACTOR
# ==========================================
def extract_text_book(input_path: str) -> str:
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf_simple(input_path)
    else:
        return ocr_image_simple(input_path)
