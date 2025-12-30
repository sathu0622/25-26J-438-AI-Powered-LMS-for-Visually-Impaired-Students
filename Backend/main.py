"""
FastAPI Backend for Document Processing
Handles OCR, Grammar Correction, Resource Type Detection, and Summarization
"""

import os
import sys
import cv2
import pytesseract
import pdf2image
from pdf2image import convert_from_path
from PIL import Image
import json
import numpy as np
import re
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from peft import PeftModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ============================================================
# CONFIGURATION
# ============================================================

# Get the base directory
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model"

# Tesseract configuration (Windows)
if sys.platform == "win32":
    # Common Windows Tesseract paths
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    else:
        print("Warning: Tesseract not found. Please install Tesseract-OCR or set pytesseract.pytesseract.tesseract_cmd")
    
    # Poppler configuration (Windows)
    poppler_path = BASE_DIR.parent / "Release-25.12.0-0" / "poppler-25.12.0" / "Library" / "bin"
    if poppler_path.exists():
        os.environ["PATH"] = str(poppler_path) + os.pathsep + os.environ.get("PATH", "")
else:
    # Linux/Mac - use system paths
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Model paths
TYPE_MODEL_PATH = MODEL_DIR / "book_magazine_newspaper_model_super_finetuned_FIXED.keras"
T5_MODEL_DIR = MODEL_DIR / "final"
GRAMMAR_MODEL_NAME = "prithivida/grammar_error_correcter_v1"

# Model configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL LOADING
# ============================================================

print("Loading models...")

# Load resource type detection model
type_model = load_model(str(TYPE_MODEL_PATH), compile=False)
print(f"✓ Resource type model loaded from {TYPE_MODEL_PATH}")

# Load grammar correction model
grammar_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_NAME)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_NAME)
grammar_model.to(DEVICE)
print(f"✓ Grammar correction model loaded ({DEVICE})")

# Load T5 summarization model with PEFT adapter
base_model_name = "google/flan-t5-base"
print(f"  Loading base model: {base_model_name}...")
summ_tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_summ_model = T5ForConditionalGeneration.from_pretrained(base_model_name)

print(f"  Loading PEFT adapter from {T5_MODEL_DIR}...")
summ_model = PeftModel.from_pretrained(base_summ_model, str(T5_MODEL_DIR))
summ_model.to(DEVICE)
summ_model.eval()
print(f"✓ T5 summarization model with PEFT adapter loaded ({DEVICE})")

print("Model loading complete!\n")

# ============================================================
# ADVANCED OCR FUNCTIONS FOR ALL LAYOUTS
# ============================================================

def detect_columns(image_np: np.ndarray) -> List[Tuple[int, int]]:
    """
    Detect column boundaries in an image using vertical projection profiling.
    Returns list of (start_x, end_x) for each column.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Sum pixels vertically
    vertical_projection = np.sum(binary, axis=0)
    
    # Smooth the projection
    kernel_size = 21
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(vertical_projection, kernel, mode='same')
    
    # Find valleys (spaces between columns)
    threshold = np.mean(smoothed) * 0.3
    valleys = smoothed < threshold
    
    # Find column boundaries
    columns = []
    in_column = False
    start = 0
    
    for i, is_valley in enumerate(valleys):
        if not in_column and not is_valley:
            # Start of column
            start = i
            in_column = True
        elif in_column and is_valley:
            # End of column
            end = i
            # Only add if column is wide enough
            if end - start > image_np.shape[1] * 0.05:  # At least 5% of width
                columns.append((start, end))
            in_column = False
    
    # Handle case where image ends in a column
    if in_column:
        columns.append((start, len(valleys)-1))
    
    # If no columns detected, assume single column
    if not columns:
        columns = [(0, image_np.shape[1]-1)]
    
    # Merge columns that are too close
    merged_columns = []
    min_gap = image_np.shape[1] * 0.02  # 2% of width
    
    for col in columns:
        if not merged_columns:
            merged_columns.append(col)
        else:
            last_start, last_end = merged_columns[-1]
            current_start, current_end = col
            
            if current_start - last_end < min_gap:
                # Merge columns
                merged_columns[-1] = (last_start, current_end)
            else:
                merged_columns.append(col)
    
    return merged_columns

def extract_text_from_region(image_np: np.ndarray, region: Tuple[int, int, int, int], 
                           config: str = "--psm 6") -> str:
    """Extract text from a specific region of image."""
    x1, y1, x2, y2 = region
    roi = image_np[y1:y2, x1:x2]
    
    # Preprocess ROI for better OCR
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing for better OCR
    denoised = cv2.medianBlur(gray_roi, 3)
    _, binary_roi = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to RGB for pytesseract
    rgb_roi = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2RGB)
    
    text = pytesseract.image_to_string(rgb_roi, config=config, lang="eng")
    return text.strip()

def detect_paragraphs_in_column(image_np: np.ndarray, column_x: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Detect paragraphs within a column using horizontal projection.
    Returns list of (start_y, end_y) for each paragraph.
    """
    x1, x2 = column_x
    column_img = image_np[:, x1:x2]
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(column_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Sum pixels horizontally
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find text lines
    threshold = np.mean(horizontal_projection) * 0.1
    text_lines = horizontal_projection > threshold
    
    # Group lines into paragraphs
    paragraphs = []
    in_paragraph = False
    start = 0
    min_paragraph_height = image_np.shape[0] * 0.01  # At least 1% of height
    
    for i, is_text in enumerate(text_lines):
        if not in_paragraph and is_text:
            start = i
            in_paragraph = True
        elif in_paragraph and not is_text:
            # Check if we've had enough consecutive non-text lines
            look_ahead = min(i + int(min_paragraph_height * 2), len(text_lines) - 1)
            if not any(text_lines[i:look_ahead]):
                end = i
                if end - start > min_paragraph_height:
                    paragraphs.append((start, end))
                in_paragraph = False
    
    # Handle last paragraph
    if in_paragraph:
        paragraphs.append((start, len(text_lines)-1))
    
    return paragraphs

def ocr_image_with_layout_analysis(image_path: str) -> str:
    """
    Extract text from image with intelligent layout analysis.
    Handles single column, multi-column, and complex layouts.
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = img.shape[:2]
        
        # Try multiple OCR strategies
        extracted_texts = []
        
        # Strategy 1: Direct OCR (for simple layouts)
        direct_text = pytesseract.image_to_string(img, lang="eng")
        if direct_text.strip():
            extracted_texts.append(("direct", direct_text))
        
        # Strategy 2: Column-based OCR (for multi-column layouts)
        columns = detect_columns(img)
        
        if len(columns) > 1:
            # Multi-column layout detected
            print(f"Detected {len(columns)} columns")
            column_texts = []
            
            for col_idx, (x1, x2) in enumerate(columns):
                # Detect paragraphs within this column
                paragraphs = detect_paragraphs_in_column(img, (x1, x2))
                
                column_content = []
                for para_idx, (y1, y2) in enumerate(paragraphs):
                    # Extract text from paragraph
                    para_region = (x1, y1, x2, y2)
                    para_text = extract_text_from_region(img, para_region, "--psm 6")
                    if para_text:
                        column_content.append(para_text)
                
                # Join paragraphs in reading order (top to bottom)
                if column_content:
                    column_texts.append("\n\n".join(column_content))
            
            # Join columns based on typical reading patterns
            # For 2 columns: left then right
            # For 3+ columns: top to bottom in each column before moving to next
            if len(column_texts) == 2:
                # Two-column layout: read left column completely, then right
                combined = f"{column_texts[0]}\n\n{column_texts[1]}"
            elif len(column_texts) > 2:
                # Multi-column: read in snake pattern (left to right, top to bottom)
                # For now, just concatenate
                combined = "\n\n".join(column_texts)
            else:
                combined = column_texts[0] if column_texts else ""
            
            extracted_texts.append(("columns", combined))
        
        # Strategy 3: Page segmentation modes
        psm_modes = {
            "psm1": "--psm 1",  # Automatic page segmentation with OSD
            "psm3": "--psm 3",  # Fully automatic page segmentation, no OSD
            "psm6": "--psm 6",  # Assume a single uniform block of text
            "psm11": "--psm 11"  # Sparse text with OSD
        }
        
        for mode_name, config in psm_modes.items():
            try:
                mode_text = pytesseract.image_to_string(img, config=config, lang="eng")
                if mode_text.strip() and len(mode_text) > 100:  # Only keep if substantial text
                    extracted_texts.append((mode_name, mode_text))
            except:
                continue
        
        # Choose the best extraction
        if not extracted_texts:
            return ""
        
        # Score each extraction
        scored_texts = []
        for method, text in extracted_texts:
            score = 0
            
            # Prefer longer text (but not too long from artifacts)
            text_len = len(text)
            if 100 < text_len < 10000:
                score += min(text_len / 1000, 5)  # Up to 5 points
            
            # Prefer text with proper sentence structure
            sentences = re.split(r'[.!?]+', text)
            avg_sentence_len = sum(len(s.strip().split()) for s in sentences if s.strip()) / max(len(sentences), 1)
            if 5 < avg_sentence_len < 25:
                score += 3
            
            # Prefer text with proper word spacing
            words = text.split()
            avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
            if 3 < avg_word_len < 10:
                score += 2
            
            # Penalize text with too many special characters
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s.,!?\-]', text))
            if len(text) > 0:
                special_ratio = special_chars / len(text)
                if special_ratio > 0.1:
                    score -= 5
            
            scored_texts.append((score, text, method))
        
        # Sort by score and pick the best
        scored_texts.sort(reverse=True, key=lambda x: x[0])
        best_text = scored_texts[0][1]
        
        print(f"Selected OCR method: {scored_texts[0][2]} with score: {scored_texts[0][0]:.2f}")
        
        return best_text
        
    except Exception as e:
        print(f"Advanced OCR error: {e}")
        # Fallback to basic OCR
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return pytesseract.image_to_string(img_rgb, lang="eng")
        except:
            return ""

def ocr_pdf_with_layout(pdf_path: str) -> str:
    """Extract text from PDF with layout analysis for each page."""
    try:
        pages = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better OCR
        text_pages = []
        
        for page_num, page in enumerate(pages):
            print(f"Processing PDF page {page_num + 1}/{len(pages)}...")
            
            # Save page as temporary image
            temp_dir = tempfile.gettempdir()
            temp_img = os.path.join(temp_dir, f"temp_page_{page_num}.png")
            
            # Save with high quality
            page.save(temp_img, "PNG", dpi=(300, 300))
            
            # Process with layout analysis
            page_text = ocr_image_with_layout_analysis(temp_img)
            text_pages.append(page_text)
            
            # Clean up
            if os.path.exists(temp_img):
                os.remove(temp_img)
        
        # Combine pages with page markers
        combined = "\n\n[PAGE BREAK]\n\n".join(text_pages)
        return combined
        
    except Exception as e:
        print(f"PDF OCR error: {e}")
        # Fallback to basic PDF OCR
        try:
            pages = convert_from_path(pdf_path)
            return " ".join([pytesseract.image_to_string(page, lang="eng") for page in pages])
        except:
            return ""

def extract_text(input_path: str) -> str:
    """Extract text from PDF or image file with layout analysis."""
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf_with_layout(input_path)
    else:
        return ocr_image_with_layout_analysis(input_path)

# ============================================================
# GRAMMAR CORRECTION
# ============================================================

def correct_text(text: str) -> str:
    """Correct grammar and OCR errors in text."""
    try:
        # Split long text into chunks to avoid token limit
        max_length = 512
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            corrected_chunks = []
            for chunk in chunks:
                inputs = grammar_tokenizer(
                    chunk, 
                    return_tensors="pt", 
                    max_length=max_length, 
                    truncation=True
                ).to(DEVICE)
                outputs = grammar_model.generate(
                    inputs['input_ids'], 
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
                corrected_chunk = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
                corrected_chunks.append(corrected_chunk)
            return " ".join(corrected_chunks)
        else:
            inputs = grammar_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True
            ).to(DEVICE)
            outputs = grammar_model.generate(
                inputs['input_ids'], 
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return text  # Return original on error

# ============================================================
# RESOURCE TYPE DETECTION
# ============================================================

def predict_resource_type(img_path: str) -> tuple[str, float]:
    """Predict the resource type (Books, Magazine, Newspapers) from an image."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = type_model.predict(arr, verbose=0)
    cls = CLASS_NAMES[np.argmax(pred)]
    conf = float(np.max(pred))
    return cls.lower(), conf

# ============================================================
# T5 SUMMARIZATION
# ============================================================

def get_prefix(type_name: str) -> str:
    """Get the summarization prefix based on resource type."""
    if type_name == "newspapers":
        return "summarize: short summary: "
    elif type_name == "magazine":
        return "summarize: medium summary: "
    elif type_name == "books":
        return "summarize: long summary in detail: "
    return "summarize: "

def summarize_text(text: str, source_type: str) -> str:
    """Summarize text using T5 model with type-specific prefixes."""
    try:
        prefix = get_prefix(source_type)
        input_text = prefix + text[:1000]  # Limit input length
        
        inputs = summ_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        ).to(DEVICE)
        
        with torch.no_grad():
            output_ids = summ_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=300,
                num_beams=4,
                early_stopping=True
            )
        
        return summ_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Summarization error: {e}")
        return text[:500] + "..." if len(text) > 500 else text  # Fallback

# ============================================================
# ARTICLE SPLITTING (FOR NEWSPAPERS)
# ============================================================

def split_into_articles(text: str) -> list[str]:
    """Split newspaper text into individual articles with improved detection."""
    # First, split by page breaks
    pages = re.split(r'\[PAGE BREAK\]', text)
    all_articles = []
    
    for page_text in pages:
        if not page_text.strip():
            continue
            
        # Multiple strategies for article detection
        
        # Strategy 1: Split by headlines (all caps, numbers, etc.)
        headline_pattern = r'(?:\n{2,}|\r\n{2,})([A-Z][A-Z0-9\s\-]{5,}(?:\n|$))'
        headline_matches = list(re.finditer(headline_pattern, page_text))
        
        # Strategy 2: Split by datelines (common in newspapers)
        dateline_pattern = r'(?:\n{2,}|\r\n{2,})([A-Z][a-z]+,?\s+\w+\s+\d{1,2}(?:,?\s+\d{4})?(?:\s+--\s+)?)'
        dateline_matches = list(re.finditer(dateline_pattern, page_text))
        
        # Strategy 3: Split by bylines
        byline_pattern = r'(?:\n{2,}|\r\n{2,})(By\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)'
        byline_matches = list(re.finditer(byline_pattern, page_text))
        
        # Combine all split points
        split_points = []
        for match in headline_matches + dateline_matches + byline_matches:
            split_points.append(match.start())
        
        split_points = sorted(set(split_points))
        
        # Split the text
        if split_points:
            last_pos = 0
            for pos in split_points:
                if pos - last_pos > 100:  # Minimum article length
                    article = page_text[last_pos:pos].strip()
                    if article:
                        all_articles.append(article)
                last_pos = pos
            
            # Add the last piece
            last_article = page_text[last_pos:].strip()
            if last_article:
                all_articles.append(last_article)
        else:
            # No clear splits found, use paragraph-based splitting
            paragraphs = re.split(r'\n{2,}', page_text)
            current_article = []
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # Check if this paragraph looks like a new article start
                if (re.match(r'^[A-Z][A-Z0-9\s\-]{5,}$', para[:50]) or  # Headline
                    re.match(r'^By\s+[A-Z]', para[:20]) or  # Byline
                    re.match(r'^[A-Z][a-z]+,?\s+\w+\s+\d', para[:30])):  # Dateline
                    
                    # Save current article if exists
                    if current_article:
                        all_articles.append(' '.join(current_article))
                        current_article = []
                
                current_article.append(para)
            
            # Add the last article
            if current_article:
                all_articles.append(' '.join(current_article))
    
    # If no articles were detected, return the whole text as one article
    if not all_articles:
        return [text]
    
    # Clean up articles (remove very short ones)
    cleaned_articles = []
    for article in all_articles:
        if len(article.split()) >= 20:  # At least 20 words
            cleaned_articles.append(article)
    
    return cleaned_articles if cleaned_articles else [text]

# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def process_document(input_path: str) -> dict:
    """Process a document through the full pipeline."""
    try:
        # Step 1: Extract raw text with layout analysis
        print("Step 1: Extracting text with layout analysis...")
        raw_text = extract_text(input_path)
        
        if not raw_text or len(raw_text.strip()) == 0:
            raise ValueError("No text extracted from document")
        
        print(f"Extracted {len(raw_text)} characters")
        
        # Step 2: Correct OCR + grammar
        print("Step 2: Correcting grammar...")
        corrected_text = correct_text(raw_text)
        
        # Step 3: Detect resource type
        print("Step 3: Detecting resource type...")
        if input_path.lower().endswith(".pdf"):
            pages = convert_from_path(input_path, first_page=1, last_page=1)
            if pages:
                temp_dir = tempfile.gettempdir()
                temp_img = os.path.join(temp_dir, "temp_detect.jpg")
                pages[0].save(temp_img, "JPEG", quality=95)
                resource_type, conf = predict_resource_type(temp_img)
                if os.path.exists(temp_img):
                    os.remove(temp_img)
            else:
                resource_type, conf = "books", 0.5
        else:
            resource_type, conf = predict_resource_type(input_path)
        
        print(f"Detected resource type: {resource_type} (confidence: {conf:.2f})")
        
        # Step 4: Split if newspaper
        print("Step 4: Processing articles...")
        if resource_type == "newspapers":
            articles = split_into_articles(corrected_text)
            print(f"Split into {len(articles)} articles")
        else:
            articles = [corrected_text]
        
        # Step 5: Generate summaries
        print("Step 5: Generating summaries...")
        summaries = []
        for i, art in enumerate(articles):
            if i < 10:  # Limit to first 10 articles to avoid excessive processing
                print(f"  Summarizing article {i+1}/{len(articles)}...")
                summary = summarize_text(art, resource_type)
                summaries.append(summary)
            else:
                summaries.append("(Additional articles not summarized)")
        
        # Step 6: Build final output
        result = {
            "resource_type": resource_type,
            "confidence": conf,
            "extracted_text": corrected_text,
            "summaries": summaries,
            "num_articles": len(articles)
        }
        
        return result
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise Exception(f"Processing error: {str(e)}")

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(title="Document Processor API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Document Processor API is running",
        "models_loaded": {
            "resource_type": type_model is not None,
            "grammar": grammar_model is not None,
            "summarization": summ_model is not None
        },
        "ocr_features": "Advanced layout analysis enabled"
    }

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded document (PDF or image)."""
    # Validate file type
    allowed_types = [
        "application/pdf",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/tiff",
        "image/bmp"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: PDF, JPEG, PNG, TIFF, BMP"
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    file_ext = os.path.splitext(file.filename)[1] or (".pdf" if file.content_type == "application/pdf" else ".jpg")
    temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{file_ext}")
    
    try:
        # Save file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"Processing file: {file.filename}")
        
        # Process document
        result = process_document(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)