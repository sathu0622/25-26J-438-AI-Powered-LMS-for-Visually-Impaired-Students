# config.py
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

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
AZURE_ENDPOINT = (os.getenv("AZURE_ENDPOINT") or "").strip()
AZURE_KEY = (os.getenv("AZURE_KEY") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "lms_db")
MONGODB_FAVORITES_COLLECTION = os.getenv("MONGODB_FAVORITES_COLLECTION", "favorite_articles")
# Optional: path to PEM CA bundle (overrides certifi for Atlas TLS)
MONGODB_TLS_CA_FILE = os.getenv("MONGODB_TLS_CA_FILE", "").strip()
# Optional: server selection timeout in ms (default 10000 for Atlas cold start / slow networks)
MONGODB_SERVER_SELECTION_TIMEOUT_MS = int(
    os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "10000")
)
# Optional: "1" only for debugging MITM/firewall issues — not for production
MONGODB_TLS_INSECURE = os.getenv("MONGODB_TLS_INSECURE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Q&A Model Configuration
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# Global storage
processed_documents = {}



# sudo apt update

# # Install Tesseract
# sudo apt install -y tesseract-ocr

# # Install Poppler
# sudo apt install -y poppler-utils



# import os
# import sys
# import torch
# import shutil
# from pathlib import Path
# from typing import Dict, Any
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# BASE_DIR = Path(__file__).parent
# MODEL_DIR = BASE_DIR / "Model"

# # =========================
# # Device configuration
# # =========================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")

# # =========================
# # Tesseract configuration
# # =========================
# if sys.platform == "win32":
#     tesseract_paths = [
#         r"C:\Program Files\Tesseract-OCR\tesseract.exe",
#         r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
#     ]
#     for path in tesseract_paths:
#         if os.path.exists(path):
#             os.environ["TESSERACT_CMD"] = path
#             break
#     else:
#         print("Warning: Tesseract not found. Please install Tesseract-OCR")

# else:
#     # Linux (Ubuntu Azure VM)
#     if shutil.which("tesseract") is None:
#         print("Warning: Tesseract is not installed on this system!")

#     if shutil.which("pdftoppm") is None:
#         print("Warning: Poppler is not installed on this system!")

# # =========================
# # Model paths
# # =========================
# TYPE_MODEL_PATH = MODEL_DIR / "book_magazine_newspaper_model_super_finetuned_FIXED.keras"
# T5_MODEL_DIR = MODEL_DIR / "final"

# # =========================
# # Configuration
# # =========================
# IMG_SIZE = (224, 224)
# CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']

# # =========================
# # Azure Document Intelligence
# # =========================
# AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
# AZURE_KEY = os.getenv("AZURE_KEY")

# # =========================
# # Q&A Model
# # =========================
# QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# # =========================
# # Global storage
# # =========================
# processed_documents = {}