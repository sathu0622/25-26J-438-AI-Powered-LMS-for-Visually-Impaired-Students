"""
FastAPI Backend for Document Processing
Simplified OCR without Multi-Column Detection and Layout Analysis
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

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model"

# Tesseract configuration
if sys.platform == "win32":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    else:
        print("Warning: Tesseract not found. Please install Tesseract-OCR")
    
    poppler_path = BASE_DIR.parent / "Release-25.12.0-0" / "poppler-25.12.0" / "Library" / "bin"
    if poppler_path.exists():
        os.environ["PATH"] = str(poppler_path) + os.pathsep + os.environ.get("PATH", "")
else:
    # Linux/Mac configuration
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Model paths
TYPE_MODEL_PATH = MODEL_DIR / "book_magazine_newspaper_model_super_finetuned_FIXED.keras"
T5_MODEL_DIR = MODEL_DIR / "final"
GRAMMAR_MODEL_NAME = "prithivida/grammar_error_correcter_v1"

# Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL LOADING
# ============================================================

print("Loading models...")

type_model = load_model(str(TYPE_MODEL_PATH), compile=False)
print(f"✓ Resource type model loaded")

grammar_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_NAME)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_NAME)
grammar_model.to(DEVICE)
print(f"✓ Grammar correction model loaded ({DEVICE})")

base_model_name = "google/flan-t5-base"
summ_tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_summ_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
summ_model = PeftModel.from_pretrained(base_summ_model, str(T5_MODEL_DIR))
summ_model.to(DEVICE)
summ_model.eval()
print(f"✓ T5 summarization model loaded ({DEVICE})")

print("Model loading complete!\n")

# ============================================================
# SIMPLIFIED OCR FUNCTIONS
# ============================================================

def ocr_image_simple(image_path: str) -> str:
    """
    Simple OCR function using basic Tesseract.
    """
    try:
        print(f"Processing image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = img.shape[:2]
        print(f"  Image size: {width}x{height}")
        
        # Convert to RGB (Tesseract needs RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run OCR with standard configuration
        text = pytesseract.image_to_string(img_rgb, lang="eng")
        
        print(f"  Extracted {len(text)} characters")
        return text.strip()
        
    except Exception as e:
        print(f"OCR Error: {e}")
        # Try fallback method
        try:
            # Try with PIL as fallback
            pil_img = Image.open(image_path)
            text = pytesseract.image_to_string(pil_img, lang="eng")
            return text.strip()
        except:
            return ""

def ocr_pdf_simple(pdf_path: str) -> str:
    """Extract text from PDF using simple OCR."""
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images with standard DPI
        print("  Converting PDF to images...")
        pages = convert_from_path(pdf_path, dpi=200)
        
        all_pages_text = []
        
        for page_num, page in enumerate(pages):
            print(f"  Processing page {page_num + 1}/{len(pages)}...")
            
            # Save page as temporary image
            temp_dir = tempfile.gettempdir()
            temp_img = os.path.join(temp_dir, f"temp_page_{page_num}_{os.urandom(4).hex()}.png")
            
            page.save(temp_img, "PNG", dpi=(200, 200))
            
            # Process with simple OCR
            page_text = ocr_image_simple(temp_img)
            
            if page_text.strip():
                all_pages_text.append(page_text)
            
            # Clean up
            try:
                os.remove(temp_img)
            except:
                pass
        
        # Combine all pages with page break markers
        combined = "\n\n[PAGE BREAK]\n\n".join(all_pages_text)
        print(f"  Total extracted characters: {len(combined)}")
        
        return combined
        
    except Exception as e:
        print(f"PDF OCR Error: {e}")
        return ""

def extract_text(input_path: str) -> str:
    """Main text extraction function using simplified OCR."""
    if input_path.lower().endswith(".pdf"):
        return ocr_pdf_simple(input_path)
    else:
        return ocr_image_simple(input_path)

# ============================================================
# GRAMMAR CORRECTION
# ============================================================

def correct_text(text: str) -> str:
    """Correct grammar and OCR errors in text."""
    try:
        max_length = 512
        # Split into sentences for better correction
        sentences = re.split(r'(?<=[.!?])\s+', text)
        corrected_sentences = []
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    inputs = grammar_tokenizer(
                        current_chunk, 
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
                    corrected = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    corrected_sentences.append(corrected)
                current_chunk = sentence + " "
        
        # Process remaining chunk
        if current_chunk:
            inputs = grammar_tokenizer(
                current_chunk, 
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
            corrected = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_sentences.append(corrected)
        
        return " ".join(corrected_sentences)
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return text

# ============================================================
# RESOURCE TYPE DETECTION
# ============================================================

def predict_resource_type(img_path: str) -> tuple[str, float]:
    """Predict resource type from image."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = type_model.predict(arr, verbose=0)
    cls = CLASS_NAMES[np.argmax(pred)]
    conf = float(np.max(pred))
    return cls.lower(), conf

# ============================================================
# SUMMARIZATION
# ============================================================

def get_prefix(type_name: str) -> str:
    """Get summarization prefix based on type."""
    if type_name == "newspapers":
        return "summarize: short summary: "
    elif type_name == "magazine":
        return "summarize: medium summary: "
    elif type_name == "books":
        return "summarize: long summary in detail: "
    return "summarize: "

def summarize_text(text: str, source_type: str) -> str:
    """Summarize text using T5 model."""
    try:
        prefix = get_prefix(source_type)
        input_text = prefix + text[:1000]
        
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
        return text[:500] + "..." if len(text) > 500 else text

# ============================================================
# ARTICLE SPLITTING
# ============================================================

def split_into_articles(text: str) -> list[str]:
    """Split newspaper text into articles."""
    # Split by column breaks and page breaks first
    sections = re.split(r'\[(?:COLUMN|PAGE) BREAK\]', text)
    
    articles = []
    for section in sections:
        if not section.strip():
            continue
        
        # Try to detect article boundaries within section
        # Headlines (all caps, short lines)
        headline_pattern = r'\n\n([A-Z][A-Z\s\-]{10,})\n'
        parts = re.split(headline_pattern, section)
        
        current_article = ""
        for i, part in enumerate(parts):
            if i % 2 == 1:  # It's a headline
                if current_article and len(current_article.split()) > 30:
                    articles.append(current_article.strip())
                current_article = part + "\n"
            else:
                current_article += part
        
        if current_article and len(current_article.split()) > 30:
            articles.append(current_article.strip())
    
    return articles if articles else [text]

# ============================================================
# MAIN PROCESSING
# ============================================================

def process_document(input_path: str) -> dict:
    """Process document through full pipeline."""
    try:
        print("\n" + "="*60)
        print("STARTING DOCUMENT PROCESSING")
        print("="*60)
        
        # Step 1: Extract text
        print("\n[1/5] Extracting text with simple OCR...")
        raw_text = extract_text(input_path)
        
        if not raw_text or len(raw_text.strip()) == 0:
            raise ValueError("No text extracted from document")
        
        print(f"✓ Extracted {len(raw_text)} characters")
        
        # Step 2: Grammar correction
        print("\n[2/5] Correcting grammar and OCR errors...")
        corrected_text = correct_text(raw_text)
        print(f"✓ Text corrected")
        
        # Step 3: Detect resource type
        print("\n[3/5] Detecting resource type...")
        if input_path.lower().endswith(".pdf"):
            pages = convert_from_path(input_path, first_page=1, last_page=1)
            if pages:
                temp_dir = tempfile.gettempdir()
                temp_img = os.path.join(temp_dir, f"temp_detect_{os.urandom(4).hex()}.jpg")
                pages[0].save(temp_img, "JPEG", quality=95)
                resource_type, conf = predict_resource_type(temp_img)
                try:
                    os.remove(temp_img)
                except:
                    pass
            else:
                resource_type, conf = "books", 0.5
        else:
            resource_type, conf = predict_resource_type(input_path)
        
        print(f"✓ Resource type: {resource_type} (confidence: {conf:.2f})")
        
        # Step 4: Split articles if newspaper
        print("\n[4/5] Processing articles...")
        if resource_type == "newspapers":
            articles = split_into_articles(corrected_text)
            print(f"✓ Split into {len(articles)} articles")
        else:
            articles = [corrected_text]
            print(f"✓ Single document")
        
        # Step 5: Generate summaries
        print("\n[5/5] Generating summaries...")
        summaries = []
        for i, art in enumerate(articles[:10]):  # Limit to 10 articles
            print(f"  Summarizing article {i+1}...")
            summary = summarize_text(art, resource_type)
            summaries.append(summary)
        
        print(f"✓ Generated {len(summaries)} summaries")
        
        result = {
            "resource_type": resource_type,
            "confidence": conf,
            "extracted_text": corrected_text,
            "summaries": summaries,
            "num_articles": len(articles),
            "text_length": len(corrected_text)
        }
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Processing error: {str(e)}")
        raise Exception(f"Processing error: {str(e)}")

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(title="Simplified Document Processor API", version="2.0.0")

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
        "message": "Simplified Document Processor API v2.0",
        "features": {
            "simple_ocr": True,
            "resource_type_detection": True,
            "grammar_correction": True,
            "summarization": True
        },
        "models_loaded": {
            "resource_type": type_model is not None,
            "grammar": grammar_model is not None,
            "summarization": summ_model is not None
        }
    }

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded document."""
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
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    temp_dir = tempfile.gettempdir()
    file_ext = os.path.splitext(file.filename)[1] or (".pdf" if file.content_type == "application/pdf" else ".jpg")
    temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{file_ext}")
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\n{'='*60}")
        print(f"Processing uploaded file: {file.filename}")
        print(f"{'='*60}")
        
        result = process_document(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("STARTING SIMPLIFIED DOCUMENT PROCESSOR SERVER")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)