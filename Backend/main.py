"""
FastAPI Backend for Document Processing
Handles PDF upload, resource type detection, OCR, and summarization
"""
import os
import re
import tempfile
from pathlib import Path
from typing import List
import json

import torch
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
import language_tool_python
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration
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

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']

# Global model variables
tokenizer = None
model = None
type_model = None
grammar_tool = None

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

def load_grammar_corrector():
    """Load language-tool for grammar correction"""
    global grammar_tool
    print("Loading language-tool for grammar correction...")
    grammar_tool = language_tool_python.LanguageTool('en-US')
    print("Grammar corrector loaded")

def load_type_detection_model():
    """Load resource type detection model"""
    global type_model
    
    print(f"Loading type detection model from {TYPE_MODEL_PATH}...")
    try:
        import tensorflow as tf
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Try different loading methods
        try:
            type_model = load_model(str(TYPE_MODEL_PATH), compile=False)
            print("✓ Type detection model loaded successfully")
            return
        except Exception as e1:
            print(f"Method 1 failed: {str(e1)[:100]}...")
        
        try:
            with tf.keras.utils.custom_object_scope({}):
                type_model = tf.keras.models.load_model(str(TYPE_MODEL_PATH), compile=False)
            print("✓ Type detection model loaded (method 2)")
            return
        except Exception as e2:
            print(f"Method 2 failed: {str(e2)[:100]}...")
        
        print("\n⚠️  WARNING: Could not load type detection model!")
        print("   Resource type detection will use a simple fallback method")
        type_model = None
        
    except Exception as e:
        print(f"Fatal error: {e}")
        type_model = None

def check_poppler_installation():
    """Check if Poppler is installed and accessible"""
    try:
        setup_poppler_path()
        if os.path.exists(os.path.join(POPPLER_BIN_PATH, "pdftoppm.exe")):
            print(f"✓ Poppler found at: {POPPLER_BIN_PATH}")
        else:
            print(f"⚠️  Poppler not found at: {POPPLER_BIN_PATH}")
    except Exception as e:
        print(f"⚠️  Poppler check warning: {e}")

def check_tesseract_installation():
    """Check if Tesseract is installed and accessible"""
    try:
        import pytesseract
        from config import TESSERACT_CMD
        
        if TESSERACT_CMD:
            print(f"[OK] Tesseract found at: {TESSERACT_CMD}")
            try:
                version = pytesseract.get_tesseract_version()
                print(f"[OK] Tesseract OCR version: {version}")
            except Exception as e:
                print(f"[WARNING] Tesseract found but may not be working: {e}")
        else:
            print("[WARNING] Tesseract not found.")
    except Exception as e:
        print(f"[WARNING] Tesseract check warning: {e}")

@app.on_event("startup")
async def startup_event():
    """Load all models on startup"""
    try:
        load_summarization_model()
        load_grammar_corrector()
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
# IMPROVED OCR Functions
# ===============================
def preprocess_image_for_ocr(img):
    """Preprocess image to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    # Method 1: Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Method 2: Otsu's thresholding
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: Noise removal
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Try all methods and return the best one
    return denoised  # Start with denoised version

def detect_columns_improved(img):
    """
    Improved column detection using multiple methods
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Method 1: Projection profile analysis
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Sum of black pixels in each column
    vertical_projection = np.sum(binary == 255, axis=0)
    
    # Find gaps (regions with few black pixels)
    threshold = np.mean(vertical_projection) * 0.3
    gaps = []
    
    for x in range(width - 1):
        if vertical_projection[x] < threshold and vertical_projection[x + 1] < threshold:
            gaps.append(x)
    
    # Group gaps
    column_boundaries = []
    if gaps:
        current_gap = [gaps[0]]
        for i in range(1, len(gaps)):
            if gaps[i] - gaps[i-1] <= 10:  # Close gaps
                current_gap.append(gaps[i])
            else:
                if len(current_gap) > 5:  # Significant gap
                    column_boundaries.append(int(np.mean(current_gap)))
                current_gap = [gaps[i]]
        
        if len(current_gap) > 5:
            column_boundaries.append(int(np.mean(current_gap)))
    
    # If no clear boundaries detected, use common layouts
    if not column_boundaries:
        # Try common newspaper layouts: 2, 3, or 4 columns
        test_layouts = [2, 3, 4]
        best_layout = 2
        best_score = 0
        
        for num_cols in test_layouts:
            col_width = width / num_cols
            score = 0
            for i in range(1, num_cols):
                x = int(i * col_width)
                # Check if this area has low text density
                region = gray[:, max(0, x-10):min(width, x+10)]
                if np.mean(region) > 200:  # Mostly white
                    score += 1
            
            if score > best_score:
                best_score = score
                best_layout = num_cols
        
        # Create boundaries for best layout
        for i in range(1, best_layout):
            column_boundaries.append(int(i * width / best_layout))
    
    return sorted(column_boundaries)

def extract_text_with_columns(img_path):
    """
    Improved OCR with column detection
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return ""
    
    height, width = img.shape[:2]
    
    # Preprocess image
    processed = preprocess_image_for_ocr(img)
    
    # Detect column boundaries
    boundaries = detect_columns_improved(img)
    
    # Add start and end boundaries
    all_boundaries = [0] + boundaries + [width]
    
    # Extract text from each column
    column_texts = []
    
    for i in range(len(all_boundaries) - 1):
        x_start = all_boundaries[i]
        x_end = all_boundaries[i + 1]
        
        # Extract column region
        col_region = processed[:, x_start:x_end]
        
        if col_region.shape[1] < 50:  # Too narrow
            continue
        
        # Apply OCR to column
        try:
            # Use appropriate PSM mode
            if i == 0 and len(all_boundaries) > 2:
                psm_mode = '6'  # Assume uniform block
            else:
                psm_mode = '4'  # Single column
            
            text = pytesseract.image_to_string(
                col_region,
                lang='eng',
                config=f'--oem 3 --psm {psm_mode}'
            )
            
            # Clean text
            text = text.strip()
            if text:
                # Remove common OCR artifacts
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines
                column_texts.append(text)
                
        except Exception as e:
            print(f"Error in column {i}: {e}")
            continue
    
    # If no columns detected or extracted, try whole image
    if not column_texts:
        try:
            text = pytesseract.image_to_string(
                processed,
                lang='eng',
                config='--oem 3 --psm 3'  # Auto page segmentation
            )
            column_texts.append(text)
        except:
            return ""
    
    # Combine columns
    combined_text = "\n\n".join(column_texts)
    
    # Post-process text
    combined_text = post_process_ocr_text(combined_text)
    
    return combined_text

def post_process_ocr_text(text):
    """Clean up OCR text"""
    if not text:
        return ""
    
    # Remove common OCR errors
    corrections = [
        (r'(\w)-\s+(\w)', r'\1\2'),  # Fix hyphenated words split across lines
        (r'\s+\.', '.'),  # Remove spaces before periods
        (r',\s{2,}', ', '),  # Fix excessive spaces after commas
        (r'i\.e\.', 'i.e.'),  # Fix common abbreviations
        (r'e\.g\.', 'e.g.'),
        (r'etc\.', 'etc.'),
        (r'\b(\w)\s+(\w)\s+(\w)\b', r'\1\2\3'),  # Fix spaced out words
    ]
    
    for pattern, replacement in corrections:
        text = re.sub(pattern, replacement, text)
    
    # Fix paragraph breaks
    paragraphs = text.split('\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            # Ensure proper sentence capitalization
            sentences = re.split(r'(?<=[.!?])\s+', para)
            fixed_sentences = []
            
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    # Capitalize first letter
                    if sent and sent[0].islower():
                        sent = sent[0].upper() + sent[1:]
                    fixed_sentences.append(sent)
            
            cleaned_para = ' '.join(fixed_sentences)
            cleaned_paragraphs.append(cleaned_para)
    
    return '\n\n'.join(cleaned_paragraphs)

def ocr_pdf_improved(pdf_path):
    """Improved PDF OCR with column detection"""
    setup_poppler_path()
    
    try:
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        error_msg = str(e).lower()
        if "poppler" in error_msg or "pdftoppm" in error_msg:
            raise ValueError(
                "Poppler is not installed or not in PATH. "
                "Please install Poppler and add it to your system PATH."
            )
        else:
            raise ValueError(f"Error converting PDF to images: {e}")
    
    all_text = []
    
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}...")
        
        # Save page as temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_img = tmp_file.name
            page.save(temp_img, "PNG")
        
        try:
            # Extract text with column detection
            page_text = extract_text_with_columns(temp_img)
            if page_text.strip():
                all_text.append(f"--- Page {i+1} ---\n{page_text}")
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_img):
                os.remove(temp_img)
    
    return "\n\n".join(all_text)

def extract_text(input_path):
    """Main text extraction function"""
    if not os.path.exists(input_path):
        raise ValueError(f"File not found: {input_path}")
    
    if input_path.lower().endswith('.pdf'):
        return ocr_pdf_improved(input_path)
    else:
        # For image files
        return extract_text_with_columns(input_path)

# ===============================
# Grammar Correction
# ===============================
def correct_grammar_python(text):
    """Correct grammar using language-tool-python"""
    if not text or len(text) < 10:
        return text
    
    try:
        # Split into manageable chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        corrected_sentences = []
        
        for sentence in sentences:
            if len(sentence) < 5:
                corrected_sentences.append(sentence)
                continue
            
            try:
                # Check and correct grammar
                matches = grammar_tool.check(sentence)
                
                if matches:
                    # Apply corrections
                    corrected = grammar_tool.correct(sentence)
                    corrected_sentences.append(corrected)
                else:
                    corrected_sentences.append(sentence)
            except:
                corrected_sentences.append(sentence)
        
        # Recombine sentences
        corrected_text = ' '.join(corrected_sentences)
        
        # Fix common issues
        corrected_text = re.sub(r'\s+([.,!?])', r'\1', corrected_text)
        corrected_text = re.sub(r'([.,!?])(\w)', r'\1 \2', corrected_text)
        
        return corrected_text.strip()
    
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return text

# ===============================
# Resource Type Detection
# ===============================
def predict_resource_type_from_pages(pdf_path):
    """Predict resource type from PDF pages"""
    # Fallback if model is not loaded
    if type_model is None:
        print("Using fallback: Analyzing PDF structure...")
        try:
            setup_poppler_path()
            pages = convert_from_path(pdf_path, dpi=200)
            num_pages = len(pages)
            
            # Simple heuristic
            if num_pages > 50:
                return "books", 0.7
            elif num_pages > 10:
                return "magazine", 0.6
            else:
                return "newspapers", 0.6
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return "books", 0.5
    
    try:
        setup_poppler_path()
        pages = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return "books", 0.5
    
    predictions = []
    confidences = []
    
    # Sample pages for efficiency
    sample_size = min(5, len(pages))
    step = max(1, len(pages) // sample_size)
    
    for i in range(0, len(pages), step):
        if i >= len(pages) or len(predictions) >= sample_size:
            break
        
        # Save page as temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            pages[i].save(temp_path, "JPEG")
        
        try:
            # Preprocess and predict
            img = image.load_img(temp_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, 0)
            
            pred = type_model.predict(img_array, verbose=0)
            cls = CLASS_NAMES[np.argmax(pred)]
            conf = float(np.max(pred))
            
            predictions.append(cls.lower())
            confidences.append(conf)
        except Exception as e:
            print(f"Error processing page {i}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not predictions:
        return "books", 0.5
    
    # Return most common prediction
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][0]
    avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == most_common])
    
    return most_common, avg_conf

def predict_resource_type(img_path):
    """Predict resource type from single image"""
    if type_model is None:
        print("Using fallback: Analyzing image...")
        from PIL import Image as PILImage
        img = PILImage.open(img_path)
        width, height = img.size
        aspect_ratio = width / height if height > 0 else 1
        
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
def build_prompt(text, source_type):
    """Build prompt for summarization"""
    text = text.strip()
    
    if len(text) < 50:
        return "summarize: Content too short or unavailable."
    
    if source_type == "newspapers":
        return f"summarize this newspaper article in 3-4 factual sentences: {text[:2000]}"
    elif source_type == "magazine":
        return f"summarize this magazine article in 4-5 sentences: {text[:2500]}"
    elif source_type == "books":
        return f"summarize this book excerpt in detail: {text[:3000]}"
    else:
        return f"summarize: {text[:2000]}"

def summarize_text(text, source_type):
    """Generate summary using T5 model"""
    prompt = build_prompt(text, source_type)
    
    max_input_length = 1024
    max_target_length = 512 if source_type == "books" else 256
    
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
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=0.8,
            temperature=0.7
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def split_into_articles(text):
    """Split newspaper text into articles"""
    # Split by common newspaper patterns
    patterns = [
        r'\n{2,}[A-Z][A-Z\s]{5,}\n{2,}',  # All caps headlines
        r'\n{2,}\d{1,2}\s+[A-Z][a-z]+\.?\s+\d{4}\n{2,}',  # Date patterns
        r'\n{2,}[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\n{2,}',  # Title case headlines
    ]
    
    for pattern in patterns:
        articles = re.split(pattern, text)
        if len(articles) > 1:
            return [art.strip() for art in articles if art.strip()]
    
    # If no pattern found, split by double newlines
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 3:
        # Group paragraphs into articles
        articles = []
        current_article = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if paragraph looks like a new article start
            if (len(para.split()) < 20 and 
                (para.isupper() or para.endswith('.')) and
                len(current_article) > 0):
                if current_article:
                    articles.append(' '.join(current_article))
                    current_article = []
            
            current_article.append(para)
        
        if current_article:
            articles.append(' '.join(current_article))
        
        if len(articles) > 1:
            return articles
    
    return [text]

# ===============================
# Main Processing Pipeline
# ===============================
def process_document(input_path):
    """Complete document processing pipeline"""
    try:
        print(f"Processing document: {input_path}")
        
        # Step 1: Extract text with improved OCR
        print("Step 1: Extracting text...")
        raw_text = extract_text(input_path)
        
        if not raw_text or len(raw_text.strip()) < 50:
            print("Warning: Little or no text extracted")
            if len(raw_text.strip()) < 10:
                raise ValueError("No readable text found in document")
        
        print(f"Extracted {len(raw_text)} characters")
        
        # Step 2: Grammar correction
        print("Step 2: Correcting grammar...")
        corrected_text = correct_grammar_python(raw_text)
        
        # Step 3: Detect resource type
        print("Step 3: Detecting resource type...")
        if input_path.lower().endswith('.pdf'):
            resource_type, confidence = predict_resource_type_from_pages(input_path)
        else:
            resource_type, confidence = predict_resource_type(input_path)
        
        print(f"Detected type: {resource_type} (confidence: {confidence:.2f})")
        
        # Step 4: Process articles
        print("Step 4: Processing articles...")
        if resource_type == "newspapers":
            articles = split_into_articles(corrected_text)
        else:
            articles = [corrected_text]
        
        print(f"Found {len(articles)} article(s)")
        
        # Step 5: Generate summaries
        print("Step 5: Generating summaries...")
        summaries = []
        
        for i, article in enumerate(articles):
            if article.strip() and len(article.strip()) > 100:
                try:
                    summary = summarize_text(article, resource_type)
                    summaries.append(summary)
                    print(f"  Generated summary {i+1}/{len(articles)}")
                except Exception as e:
                    print(f"  Error summarizing article {i+1}: {e}")
                    summaries.append("Could not generate summary for this article.")
        
        # Step 6: Prepare result
        print("Step 6: Preparing results...")
        result = {
            "resource_type": resource_type,
            "confidence": round(float(confidence), 4),
            "extracted_text": corrected_text[:5000] + "..." if len(corrected_text) > 5000 else corrected_text,
            "full_text_length": len(corrected_text),
            "num_articles": len(articles),
            "summaries": summaries,
            "status": "success"
        }
        
        print("Processing completed successfully!")
        return result
        
    except Exception as e:
        print(f"Processing error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "resource_type": "unknown",
            "confidence": 0.0,
            "extracted_text": "",
            "num_articles": 0,
            "summaries": []
        }

# ===============================
# API Endpoints
# ===============================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Document Processing API is running",
        "device": str(DEVICE),
        "models_loaded": tokenizer is not None and model is not None
    }

@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "summarization_model": tokenizer is not None and model is not None,
        "type_model": type_model is not None,
        "grammar_tool": grammar_tool is not None
    }

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded PDF or image file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_path = tmp_file.name
        content = await file.read()
        tmp_file.write(content)
    
    try:
        # Process document
        result = process_document(tmp_path)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)