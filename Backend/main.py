"""
FastAPI Backend for Document Processing
Handles OCR, Grammar Correction, Resource Type Detection, and Summarization
"""

import os
import sys
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
import numpy as np
import re
import tempfile
from pathlib import Path
from typing import Optional

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
# OCR FUNCTIONS
# ============================================================

def ocr_image(image_path: str) -> str:
    """Extract text from an image using OCR."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_text = pytesseract.image_to_string(img_rgb, lang="eng")
        return raw_text
    except Exception as e:
        raise Exception(f"OCR error: {str(e)}")

def ocr_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using OCR."""
    try:
        pages = convert_from_path(pdf_path)
        text_pages = []
        temp_dir = tempfile.gettempdir()
        
        for i, page in enumerate(pages):
            temp_img = os.path.join(temp_dir, f"temp_page_{i}.png")
            page.save(temp_img, "PNG")
            text_pages.append(ocr_image(temp_img))
            if os.path.exists(temp_img):
                os.remove(temp_img)
        
        return " ".join(text_pages)
    except Exception as e:
        raise Exception(f"PDF OCR error: {str(e)}")

def extract_text(input_path: str) -> str:
    """Extract text from PDF or image file."""
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf(input_path)
    else:
        return ocr_image(input_path)

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
    """Split newspaper text into individual articles."""
    paragraphs = re.split(r'\n{1,2}', text)
    blocks = []
    current_block = ""
    
    for p in paragraphs:
        if len(p.strip()) == 0:
            continue
        # Check if paragraph starts with uppercase letters/numbers (likely article header)
        if re.match(r'^[A-Z0-9]{2,}', p.strip()):
            if current_block:
                blocks.append(current_block.strip())
            current_block = p.strip()
        else:
            current_block += " " + p.strip()
    
    if current_block:
        blocks.append(current_block.strip())
    
    # If no articles found, return the whole text as one article
    if not blocks:
        return [text]
    
    return blocks

# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def process_document(input_path: str) -> dict:
    """Process a document through the full pipeline."""
    try:
        # Step 1: Extract raw text
        print("Step 1: Extracting text...")
        raw_text = extract_text(input_path)
        
        if not raw_text or len(raw_text.strip()) == 0:
            raise ValueError("No text extracted from document")
        
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
                pages[0].save(temp_img, "JPEG")
                resource_type, conf = predict_resource_type(temp_img)
                if os.path.exists(temp_img):
                    os.remove(temp_img)
            else:
                resource_type, conf = "books", 0.5
        else:
            resource_type, conf = predict_resource_type(input_path)
        
        # Step 4: Split if newspaper
        print("Step 4: Processing articles...")
        if resource_type == "newspapers":
            articles = split_into_articles(corrected_text)
        else:
            articles = [corrected_text]
        
        # Step 5: Generate summaries
        print("Step 5: Generating summaries...")
        summaries = []
        for i, art in enumerate(articles):
            print(f"  Summarizing article {i+1}/{len(articles)}...")
            summary = summarize_text(art, resource_type)
            summaries.append(summary)
        
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
        }
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
        
        # Process document
        result = process_document(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
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

