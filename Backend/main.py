"""
FastAPI Backend for Document Processing
Handles PDF upload, resource type detection, OCR, and summarization
"""
import os
import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from peft import PeftModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import config for Tesseract setup
import config

# Initialize FastAPI app
app = FastAPI(title="Document Processing API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Configuration
# ===============================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model"
BASE_MODEL_NAME = "google/flan-t5-base"
ADAPTER_PATH = MODEL_DIR / "final"
TYPE_MODEL_PATH = MODEL_DIR / "book_magazine_newspaper_model_super_finetuned2.keras"
GRAMMAR_MODEL_NAME = "prithivida/grammar_error_correcter_v1"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']

# Global model variables
tokenizer = None
model = None
grammar_tokenizer = None
grammar_model = None
type_model = None

# ===============================
# Model Loading Functions
# ===============================
def load_summarization_model():
    """Load T5 model with LoRA adapter"""
    global tokenizer, model
    
    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_NAME)
    
    print(f"Loading base model from {BASE_MODEL_NAME}...")
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    
    model = model.to(DEVICE)
    model.eval()
    print(f"Summarization model loaded on {DEVICE}")

def load_grammar_model():
    """Load grammar correction model"""
    global grammar_tokenizer, grammar_model
    
    print(f"Loading grammar model: {GRAMMAR_MODEL_NAME}...")
    grammar_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_NAME)
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_NAME)
    grammar_model = grammar_model.to(DEVICE)
    grammar_model.eval()
    print("Grammar model loaded")

def load_type_detection_model():
    """Load resource type detection model"""
    global type_model
    
    print(f"Loading type detection model from {TYPE_MODEL_PATH}...")
    try:
        import tensorflow as tf
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Method 1: Try standard load with compile=False
        try:
            type_model = load_model(str(TYPE_MODEL_PATH), compile=False)
            print("✓ Type detection model loaded successfully")
            return
        except Exception as e1:
            print(f"Method 1 failed: {str(e1)[:100]}...")
        
        # Method 2: Try with tf.keras and custom_object_scope
        try:
            with tf.keras.utils.custom_object_scope({}):
                type_model = tf.keras.models.load_model(str(TYPE_MODEL_PATH), compile=False)
            print("✓ Type detection model loaded (method 2)")
            return
        except Exception as e2:
            print(f"Method 2 failed: {str(e2)[:100]}...")
        
        # Method 3: Try loading weights only (if model architecture can be inferred)
        try:
            # Try to load just the weights
            import h5py
            with h5py.File(str(TYPE_MODEL_PATH), 'r') as f:
                # Check if it's a weights-only file
                if 'model_weights' in f.keys():
                    print("Attempting to load weights only...")
                    # This would require knowing the model architecture
                    raise NotImplementedError("Weights-only loading requires model architecture")
        except Exception as e3:
            print(f"Method 3 failed: {str(e3)[:100]}...")
        
        # Method 4: Try with legacy Keras format
        try:
            # Disable eager execution temporarily (if using TF 1.x compatibility)
            type_model = tf.keras.models.load_model(
                str(TYPE_MODEL_PATH),
                compile=False,
                custom_objects=None
            )
            print("✓ Type detection model loaded (method 4)")
            return
        except Exception as e4:
            print(f"Method 4 failed: {str(e4)[:100]}...")
        
        # If all methods fail, set model to None and use fallback
        print("\n⚠️  WARNING: Could not load type detection model!")
        print("   The model file may be incompatible with TensorFlow 2.20.0")
        print("   Resource type detection will use a simple fallback method")
        print("   (based on file characteristics instead of ML model)")
        type_model = None
        
    except Exception as e:
        print(f"Fatal error: {e}")
        type_model = None

def check_poppler_installation():
    """Check if Poppler is installed and accessible"""
    try:
        setup_poppler_path()
        # Try to verify Poppler is accessible
        if os.path.exists(os.path.join(POPPLER_BIN_PATH, "pdftoppm.exe")):
            print(f"✓ Poppler found at: {POPPLER_BIN_PATH}")
        else:
            print(f"⚠️  Poppler not found at: {POPPLER_BIN_PATH}")
            print("   If you get 'Unable to get page count' error, check Poppler installation")
    except Exception as e:
        print(f"⚠️  Poppler check warning: {e}")

def check_tesseract_installation():
    """Check if Tesseract is installed and accessible"""
    try:
        import pytesseract
        from config import TESSERACT_CMD
        
        if TESSERACT_CMD:
            print(f"[OK] Tesseract found at: {TESSERACT_CMD}")
            # Try to get Tesseract version to verify it's working
            try:
                version = pytesseract.get_tesseract_version()
                print(f"[OK] Tesseract OCR version: {version}")
            except Exception as e:
                print(f"[WARNING] Tesseract found but may not be working: {e}")
                print("   Make sure Tesseract is properly installed")
        else:
            print("[WARNING] Tesseract not found. Please install Tesseract OCR:")
            print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("   Or set TESSERACT_CMD environment variable")
    except Exception as e:
        print(f"[WARNING] Tesseract check warning: {e}")

@app.on_event("startup")
async def startup_event():
    """Load all models on startup"""
    try:
        load_summarization_model()
        load_grammar_model()
        load_type_detection_model()
        check_poppler_installation()
        check_tesseract_installation()
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# ===============================
# Poppler Configuration
# ===============================
POPPLER_BIN_PATH = r"D:\Research dataset\PP1_Setup\Release-25.12.0-0\poppler-25.12.0\Library\bin"

def setup_poppler_path():
    """Add Poppler bin folder to PATH if not already present"""
    if os.path.exists(POPPLER_BIN_PATH) and os.path.exists(os.path.join(POPPLER_BIN_PATH, "pdftoppm.exe")):
        current_path = os.environ.get("PATH", "")
        if POPPLER_BIN_PATH not in current_path:
            os.environ["PATH"] = current_path + os.pathsep + POPPLER_BIN_PATH
            print(f"✓ Poppler path added: {POPPLER_BIN_PATH}")

# ===============================
# OCR Functions - Simple Approach (matching original working code)
# ===============================
def ocr_image_simple(image_path: str) -> str:
    """
    SMART COLUMN DETECTION + PHYSICAL SPLITTING
    Uses word positions to detect actual column boundaries, then splits physically
    Returns clean text WITHOUT any markers
    """
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    print(f"Image size: {img_width}x{img_height}")
    
    # Step 1: Get word positions to detect actual column boundaries
    try:
        data = pytesseract.image_to_data(img_rgb, lang="eng", output_type=pytesseract.Output.DICT)
        
        # Collect word x-positions
        word_x_positions = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text and int(data['conf'][i]) > 0:
                x = data['left'][i]
                w = data['width'][i]
                word_x_positions.append(x + w/2)  # Center of word
        
        if len(word_x_positions) < 10:
            # Not enough words, use simple extraction
            print("⚠️ Not enough words detected, using simple extraction")
            text = pytesseract.image_to_string(img_rgb, lang="eng")
            text = re.sub(r'COLUMN\s+\d+.*?layout', '', text, flags=re.IGNORECASE)
            return text.strip()
        
        # Step 2: Find column boundaries by analyzing x-position distribution
        word_x_sorted = sorted(word_x_positions)
        
        # Find gaps (column separators)
        gaps = []
        for i in range(len(word_x_sorted) - 1):
            gap = word_x_sorted[i+1] - word_x_sorted[i]
            if gap > img_width * 0.08:  # Gap > 8% of page width = likely column separator
                gaps.append((word_x_sorted[i], word_x_sorted[i+1], gap))
        
        # Get column boundaries from significant gaps
        if len(gaps) >= 1:
            # Use detected boundaries
            boundaries = [0]
            for gap_start, gap_end, gap_size in sorted(gaps, key=lambda g: g[2], reverse=True)[:4]:
                boundary = (gap_start + gap_end) / 2
                boundaries.append(int(boundary))
            boundaries.append(img_width)
            boundaries = sorted(set(boundaries))
            print(f"✓ Detected {len(boundaries)-1} columns from word positions")
        else:
            # Fallback: divide into 4 equal columns (most common)
            boundaries = [0, int(img_width*0.25), int(img_width*0.5), int(img_width*0.75), img_width]
            print(f"⚠️ No clear boundaries, using 4 equal columns")
        
        # Step 3: Extract each column physically
        columns_text = []
        for col_idx in range(len(boundaries) - 1):
            x_start = boundaries[col_idx]
            x_end = boundaries[col_idx + 1]
            
            # Extract this column
            col_img = img_rgb[:, x_start:x_end]
            
            if col_img.size == 0 or col_img.shape[1] < 50:
                continue
            
            try:
                # OCR with PSM 4 (single column)
                col_text = pytesseract.image_to_string(
                    col_img, 
                    lang="eng", 
                    config=r'--oem 3 --psm 4'
                )
                col_text = col_text.strip()
                
                # Remove any column markers
                col_text = re.sub(r'COLUMN\s+\d+.*?layout', '', col_text, flags=re.IGNORECASE)
                col_text = re.sub(r'\d+-column layout', '', col_text, flags=re.IGNORECASE)
                col_text = re.sub(r'six-column layout', '', col_text, flags=re.IGNORECASE)
                col_text = col_text.strip()
                
                if col_text and len(col_text) > 20:
                    # Clean up
                    col_text = re.sub(r'\s+', ' ', col_text)
                    col_text = re.sub(r'\n{3,}', '\n\n', col_text)
                    columns_text.append(col_text)
                    print(f"  ✓ Column {col_idx+1}: {len(col_text)} chars")
            except Exception as e:
                print(f"  ✗ Column {col_idx+1}: Error - {e}")
                continue
        
        # Return clean text (no markers)
        if len(columns_text) > 1:
            result = "\n\n".join(columns_text)
            print(f"\n✓ SUCCESS: Extracted {len(columns_text)} columns")
            return result
        elif columns_text:
            return columns_text[0]
        else:
            # Fallback
            text = pytesseract.image_to_string(img_rgb, lang="eng")
            text = re.sub(r'COLUMN\s+\d+.*?layout', '', text, flags=re.IGNORECASE)
            return text.strip()
            
    except Exception as e:
        print(f"Error: {e}")
        # Final fallback
        text = pytesseract.image_to_string(img_rgb, lang="eng")
        text = re.sub(r'COLUMN\s+\d+.*?layout', '', text, flags=re.IGNORECASE)
        return text.strip()

def ocr_pdf_simple(pdf_path: str) -> str:
    """Simple PDF OCR extraction with column-aware processing"""
    setup_poppler_path()
    
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        error_msg = str(e).lower()
        if "poppler" in error_msg or "pdftoppm" in error_msg:
            raise ValueError(
                "Poppler is not installed or not in PATH. "
                "Please install Poppler:\n"
                "Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "Extract and add the 'bin' folder to your system PATH."
            )
        else:
            raise ValueError(f"Error converting PDF to images: {e}")
    
    text_pages = []
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}...")
        # Save page as temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_img = tmp_file.name
            page.save(temp_img, "PNG")
        
        try:
            # Use column-aware OCR
            page_text = ocr_image_simple(temp_img)
            text_pages.append(page_text)
        finally:
            # Clean up temp file
            if os.path.exists(temp_img):
                os.remove(temp_img)
    
    # Join pages with double newline to separate pages
    return "\n\n".join(text_pages)

def detect_columns_improved(img):
    """
    Improved column detection using multiple methods
    Returns list of x-coordinates where columns are separated
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Method 1: Detect vertical white spaces (gaps between columns)
    # Columns typically have white space between them
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # Find white areas
    
    # Project vertical: count white pixels in each column
    vertical_projection = np.sum(binary == 255, axis=0)
    
    # Find regions with high white pixel count (gaps between columns)
    threshold = np.mean(vertical_projection) * 0.7  # 70% of average
    gap_regions = []
    in_gap = False
    gap_start = 0
    
    for x in range(width):
        if vertical_projection[x] > threshold:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                gap_end = x
                gap_width = gap_end - gap_start
                # Only consider gaps that are wide enough (at least 20 pixels)
                if gap_width > 20:
                    gap_center = (gap_start + gap_end) // 2
                    gap_regions.append(gap_center)
                in_gap = False
    
    # Method 2: Detect vertical lines (if Method 1 doesn't work well)
    if len(gap_regions) < 2:
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(50, height // 10)))
        vertical_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > height * 0.2 and w < width * 0.05:
                gap_regions.append(x + w // 2)
    
    # Clean up: sort and remove duplicates
    gap_regions = sorted(set(gap_regions))
    filtered_gaps = []
    for gap in gap_regions:
        if not filtered_gaps or abs(gap - filtered_gaps[-1]) > 30:
            filtered_gaps.append(gap)
    
    # Method 3: If still no clear columns, use equal division
    # Try different column counts and pick the one that makes sense
    if len(filtered_gaps) < 2:
        # Common newspaper layouts: 3, 4, or 5 columns
        for num_cols in [5, 4, 3]:
            col_width = width / num_cols
            # Check if this division makes sense by checking text density
            test_gaps = [int(i * col_width) for i in range(1, num_cols)]
            filtered_gaps = test_gaps
            break
    
    return filtered_gaps

def extract_column_text(img, x_start, x_end, y_start=0, y_end=None):
    """Extract text from a specific column region with strict isolation"""
    if y_end is None:
        y_end = img.shape[0]
    
    # Extract column region with NO padding to avoid overlap
    # This ensures columns don't bleed into each other
    x_start_actual = max(0, int(x_start))
    x_end_actual = min(img.shape[1], int(x_end))
    
    # Make sure we have valid dimensions
    if x_end_actual <= x_start_actual:
        return ""
    
    col_img = img[y_start:y_end, x_start_actual:x_end_actual]
    
    if col_img.size == 0 or col_img.shape[0] < 50 or col_img.shape[1] < 50:
        return ""
    
    # Use PSM 4 (single uniform block of text) for column extraction
    # PSM 4 works best for single columns
    try:
        # Convert to RGB for pytesseract
        col_img_rgb = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
        
        # Try PSM 4 first (single column)
        text = pytesseract.image_to_string(col_img_rgb, lang="eng", config=r'--oem 3 --psm 4')
        
        # Clean up the text
        text = text.strip()
        # Remove excessive whitespace but keep paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
        
        return text
    except Exception as e:
        print(f"Error extracting column text: {e}")
        return ""

def ocr_image_column_aware(image_path: str) -> str:
    """Extract text from image with proper column handling"""
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # SIMPLIFIED APPROACH: Use equal-width column division
    # Most newspapers have equal-width columns, so this is more reliable
    height, width = img.shape[:2]
    
    # Try different column counts
    for num_cols in [5, 4, 6, 3]:
        columns_text = []
        col_width = width / num_cols
        
        for i in range(num_cols):
            x_start = int(i * col_width)
            x_end = int((i + 1) * col_width)
            
            # Extract this column with NO overlap
            col_img = img_bgr[:, x_start:x_end]
            
            if col_img.size == 0 or col_img.shape[1] < 80:
                continue
            
            try:
                col_img_rgb = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
                col_text = pytesseract.image_to_string(col_img_rgb, lang="eng", config=r'--oem 3 --psm 4')
                col_text = col_text.strip()
                
                if col_text and len(col_text) > 50:
                    col_text = re.sub(r'[ \t]+', ' ', col_text)
                    col_text = re.sub(r'\n{3,}', '\n\n', col_text)
                    columns_text.append(col_text)
            except:
                continue
        
        # If we got good results, use this
        if len(columns_text) >= 2:
            return "\n\n".join(
                [f"{'='*70}\nCOLUMN {i+1} ({num_cols}-column layout):\n{'='*70}\n\n{col}" 
                 for i, col in enumerate(columns_text)]
            )
    
    # Fallback
    return pytesseract.image_to_string(img_rgb, lang="eng", config=r'--oem 3 --psm 4')

def ocr_image(image_path: str) -> str:
    """Extract text from image using Tesseract with column awareness"""
    return ocr_image_column_aware(image_path)

def extract_text_from_pdf_column_aware(pdf_path: str) -> str:
    """
    Extract text from PDF with proper column handling for newspapers
    Uses column detection to extract text column by column
    """
    # Setup Poppler path
    setup_poppler_path()
    
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        error_msg = str(e).lower()
        if "poppler" in error_msg or "pdftoppm" in error_msg or "unable to get page count" in error_msg:
            raise ValueError(
                "Poppler is not installed or not in PATH. "
                "Please install Poppler:\n"
                "Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "Extract and add the 'bin' folder to your system PATH."
            )
        else:
            raise ValueError(f"Error converting PDF to images: {e}")
    
    all_text_pages = []
    
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}...")
        
        # Convert PIL to OpenCV format
        img_array = np.array(page.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Use column-aware extraction
        page_text = ocr_image_column_aware_from_array(img_cv)
        
        if not page_text.strip():
            # Fallback: try standard extraction
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            try:
                page_text = pytesseract.image_to_string(img_rgb, lang="eng", config=r'--oem 3 --psm 4')
            except:
                page_text = pytesseract.image_to_string(img_rgb, lang="eng")
        
        all_text_pages.append(page_text)
    
    return "\n\n".join(all_text_pages)

def ocr_image_column_aware_from_array(img_bgr):
    """Extract text from OpenCV image array with column awareness - SIMPLIFIED APPROACH"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_bgr.shape[:2]
    
    # SIMPLIFIED APPROACH: Divide into equal columns (most newspapers have equal-width columns)
    # Try different column counts and pick the best one
    best_result = None
    best_num_cols = 0
    
    # Try 3, 4, 5, or 6 columns (common newspaper layouts)
    for num_cols in [5, 4, 6, 3]:
        columns_text = []
        col_width = width / num_cols
        
        print(f"Trying {num_cols} columns (width per column: {col_width:.0f}px)")
        
        for i in range(num_cols):
            x_start = int(i * col_width)
            x_end = int((i + 1) * col_width)
            
            # Extract this column with NO overlap
            col_img = img_bgr[:, x_start:x_end]
            
            if col_img.size == 0 or col_img.shape[1] < 80:
                continue
            
            try:
                col_img_rgb = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)
                # Use PSM 4 for single column extraction
                col_text = pytesseract.image_to_string(col_img_rgb, lang="eng", config=r'--oem 3 --psm 4')
                col_text = col_text.strip()
                
                if col_text and len(col_text) > 50:  # Only if meaningful text
                    # Clean up
                    col_text = re.sub(r'[ \t]+', ' ', col_text)
                    col_text = re.sub(r'\n{3,}', '\n\n', col_text)
                    columns_text.append(col_text)
                    print(f"  Column {i+1}: {len(col_text)} chars")
            except Exception as e:
                print(f"  Column {i+1}: Error - {e}")
                continue
        
        # If we got good results (at least 2 columns with text), use this
        if len(columns_text) >= 2:
            best_result = columns_text
            best_num_cols = num_cols
            print(f"✓ Found {len(columns_text)} columns with text using {num_cols}-column layout")
            break
    
    # If we found columns, combine them
    if best_result and len(best_result) > 1:
        result = "\n\n".join(
            [f"{'='*70}\nCOLUMN {i+1} ({best_num_cols}-column layout):\n{'='*70}\n\n{col}" 
             for i, col in enumerate(best_result)]
        )
        return result
    elif best_result:
        return best_result[0]
    else:
        # Final fallback: use PSM 4 on whole image
        print("Using fallback: PSM 4 on whole image")
        return pytesseract.image_to_string(img_rgb, lang="eng", config=r'--oem 3 --psm 4')

def extract_text(input_path: str) -> str:
    """Extract text from PDF or image - using simple approach (matching original code)"""
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf_simple(input_path)
    else:
        return ocr_image_simple(input_path)

# ===============================
# Grammar Correction
# ===============================
def correct_text(text: str) -> str:
    """Correct grammar and OCR errors"""
    if not text.strip():
        return text
    
    try:
        # Split long text into chunks to avoid token limit
        max_chunk_length = 500
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        corrected_chunks = []
        
        for chunk in chunks:
            inputs = grammar_tokenizer(
                chunk, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = grammar_model.generate(
                    inputs['input_ids'], 
                    max_length=1024,
                    num_beams=2,
                    early_stopping=True
                )
            
            corrected_chunk = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_chunks.append(corrected_chunk)
        
        return " ".join(corrected_chunks)
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return text  # Return original if correction fails

# ===============================
# Resource Type Detection
# ===============================
def predict_resource_type_from_pages(pdf_path: str) -> tuple:
    """
    Predict resource type using all pages of the PDF
    Returns the most common prediction across all pages
    """
    # Fallback if model is not loaded
    if type_model is None:
        print("Using fallback: Analyzing PDF structure...")
        try:
            setup_poppler_path()
            pages = convert_from_path(pdf_path, dpi=200)
            # Simple heuristic: count pages and analyze layout
            num_pages = len(pages)
            if num_pages > 50:
                return "books", 0.7
            elif num_pages > 10:
                return "magazine", 0.6
            else:
                return "newspapers", 0.6
        except Exception as e:
            error_msg = str(e).lower()
            if "poppler" in error_msg or "pdftoppm" in error_msg:
                print("⚠️  Poppler not found. Using default resource type.")
                return "books", 0.5
            raise
    
    try:
        setup_poppler_path()
        pages = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        error_msg = str(e).lower()
        if "poppler" in error_msg or "pdftoppm" in error_msg:
            print("⚠️  Poppler not found. Using fallback resource type detection.")
            return "books", 0.5
        raise
    predictions = []
    confidences = []
    
    for i, page in enumerate(pages):
        # Save page as temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            page.save(temp_path, "JPEG")
        
        try:
            # Preprocess image
            img = image.load_img(temp_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, 0)
            
            # Predict
            pred = type_model.predict(img_array, verbose=0)
            cls = CLASS_NAMES[np.argmax(pred)]
            conf = float(np.max(pred))
            
            predictions.append(cls.lower())
            confidences.append(conf)
        except Exception as e:
            print(f"Error processing page {i}: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not predictions:
        return "books", 0.5  # Default
    
    # Return most common prediction with average confidence
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][0]
    avg_confidence = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == most_common])
    
    return most_common, avg_confidence

def predict_resource_type(img_path: str) -> tuple:
    """Predict resource type from single image"""
    # Fallback if model is not loaded
    if type_model is None:
        print("Using fallback: Analyzing image characteristics...")
        # Simple heuristic based on image dimensions
        from PIL import Image as PILImage
        img = PILImage.open(img_path)
        width, height = img.size
        aspect_ratio = width / height if height > 0 else 1
        
        # Newspapers are usually wider, books taller
        if aspect_ratio > 1.5:
            return "newspapers", 0.6
        elif aspect_ratio < 0.8:
            return "books", 0.6
        else:
            return "magazine", 0.6
    
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        pred = type_model.predict(img_array, verbose=0)
        cls = CLASS_NAMES[np.argmax(pred)]
        conf = float(np.max(pred))
        
        return cls.lower(), float(conf)
    except Exception as e:
        print(f"Type detection error: {e}")
        return "books", 0.5

# ===============================
# Summarization
# ===============================
def build_prompt(text: str, source_type: str) -> str:
    """Build prompt based on source type"""
    text = text.strip()
    
    if "purchase a subscription" in text.lower() or len(text) < 50:
        return "summarize: The article content is unavailable. Provide a 2-sentence generic summary."
    
    if source_type == "newspapers":
        return f"summarize newspaper article in 3-4 factual sentences: {text}"
    elif source_type == "magazine":
        return f"summarize magazine article in about half the original length with key details: {text}"
    elif source_type == "books":
        return f"summarize book excerpt in detail preserving key ideas and context: {text}"
    else:
        return f"summarize: {text}"

def summarize_text(text: str, source_type: str) -> str:
    """Generate summary using T5 model with LoRA adapter"""
    prompt = build_prompt(text, source_type)
    
    max_input_length = 1024
    max_target_length = 768 if source_type == "books" else 300
    
    inputs = tokenizer(
        prompt, 
        max_length=max_input_length, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_target_length,
            num_beams=6 if source_type == "books" else 4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            do_sample=False
        )
    
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_summary

# ===============================
# Article Splitting (for newspapers)
# ===============================
def split_into_articles(text: str) -> List[str]:
    """Split newspaper text into individual articles"""
    paragraphs = re.split(r'\n{1,2}', text)
    blocks = []
    current_block = ""
    
    for p in paragraphs:
        if len(p.strip()) == 0:
            continue
        
        # Check if paragraph starts with uppercase letters (likely article header)
        if re.match(r'^[A-Z0-9]{2,}', p.strip()):
            if current_block:
                blocks.append(current_block.strip())
            current_block = p.strip()
        else:
            current_block += " " + p.strip()
    
    if current_block:
        blocks.append(current_block.strip())
    
    return blocks if blocks else [text]

# ===============================
# Main Processing Pipeline
# ===============================
def process_document(input_path: str) -> dict:
    """Complete document processing pipeline"""
    try:
        # Step 1: Extract raw text
        print("Extracting text...")
        raw_text = extract_text(input_path)
        
        if not raw_text.strip():
            raise ValueError("No text extracted from document")
        
        # Step 2: Correct OCR + grammar
        print("Correcting grammar...")
        corrected_text = correct_text(raw_text)
        
        # Step 3: Detect resource type
        print("Detecting resource type...")
        if input_path.lower().endswith(".pdf"):
            resource_type, conf = predict_resource_type_from_pages(input_path)
        else:
            resource_type, conf = predict_resource_type(input_path)
        
        print(f"Detected type: {resource_type} (confidence: {conf:.2f})")
        
        # Step 4: Split if newspaper
        print("Processing articles...")
        if resource_type == "newspapers":
            articles = split_into_articles(corrected_text)
        else:
            articles = [corrected_text]
        
        # Step 5: Generate summaries
        print("Generating summaries...")
        summaries = []
        for i, art in enumerate(articles):
            if art.strip():
                summary = summarize_text(art, resource_type)
                summaries.append(summary)
                print(f"Generated summary {i+1}/{len(articles)}")
        
        # Step 6: Final output
        final_output = {
            "resource_type": resource_type,
            "confidence": round(conf, 4),
            "extracted_text": corrected_text,
            "num_articles": len(articles),
            "summaries": summaries
        }
        
        return final_output
        
    except Exception as e:
        print(f"Processing error: {e}")
        raise

# ===============================
# API Endpoints
# ===============================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Document Processing API is running", "device": str(DEVICE)}

@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": all([
            tokenizer is not None,
            model is not None,
            grammar_model is not None,
            type_model is not None
        ])
    }

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded PDF or image file"""
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_path = tmp_file.name
        content = await file.read()
        tmp_file.write(content)
    
    try:
        # Process document
        result = process_document(tmp_path)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

