# config.py
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Tesseract configuration
if sys.platform == "win32":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            os.environ["TESSERACT_CMD"] = path
            break
    else:
        print("Warning: Tesseract not found. Please install Tesseract-OCR")
    
    poppler_path = BASE_DIR.parent / "Release-25.12.0-0" / "poppler-25.12.0" / "Library" / "bin"
    if poppler_path.exists():
        os.environ["PATH"] = str(poppler_path) + os.pathsep + os.environ.get("PATH", "")

# Model paths
TYPE_MODEL_PATH = MODEL_DIR / "book_magazine_newspaper_model_super_finetuned_FIXED.keras"
T5_MODEL_DIR = MODEL_DIR / "final"

# Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']

# Azure Document Intelligence Configuration
AZURE_ENDPOINT = ""
AZURE_KEY = ""

# Q&A Model Configuration
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# Global storage
processed_documents = {}