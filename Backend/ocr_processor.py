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
    """
    Detect chapters and structure book text properly.
    """

    if not full_text.strip():
        return {"book_title": "Unknown Title", "chapters": []}

    # -----------------------------------------
    # 1️⃣ Clean OCR text
    # -----------------------------------------
    text = re.sub(r'\r', '\n', full_text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix OCR broken words 
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # -----------------------------------------
    # 2️⃣ Detect Chapter Headings
    # -----------------------------------------
    chapter_pattern = re.compile(
    r'^(?:THE\s+\w+\s+CHAPTER.*|[IVXLCDM]+\.\s+.+)$',
    re.IGNORECASE | re.MULTILINE
)

    matches = list(chapter_pattern.finditer(text))

    chapters = []

    # -----------------------------------------
    # 3️⃣ Extract Chapter Content
    # -----------------------------------------
    for i, match in enumerate(matches):

        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        chapter_block = text[start:end].strip()

        lines = chapter_block.split("\n")

        heading = lines[0].strip()

        content = "\n".join(lines[1:]).strip()

        # Split paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]

        chapters.append({
            "article_id": f"chapter_{i+1}",
            "heading": heading,
            "subheading": "",
            "body": paragraphs,
            "full_text": "\n\n".join(paragraphs)
        })

    # -----------------------------------------
    # 4️⃣ Detect Book Title
    # -----------------------------------------
    first_lines = text.split("\n")[:5]

    book_title = "Unknown Title"
    for line in first_lines:
        if 3 < len(line.split()) < 10 and line.isupper():
            book_title = line.strip()
            break

    print(f"Detected {len(chapters)} chapters")

    return {
        "book_title": book_title,
        "chapters": chapters
    }

# ==========================================
# 🔹 MAIN BOOK TEXT EXTRACTOR
# ==========================================
def extract_text_book(input_path: str) -> str:
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf_simple(input_path)
    else:
        return ocr_image_simple(input_path)
