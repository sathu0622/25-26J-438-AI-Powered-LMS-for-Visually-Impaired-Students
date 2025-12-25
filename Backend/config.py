"""
Configuration file for the application
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Tesseract configuration
# Common Tesseract installation paths
DEFAULT_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
]

# Try to find Tesseract automatically
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

if not TESSERACT_CMD:
    # Try default paths
    for path in DEFAULT_TESSERACT_PATHS:
        if os.path.exists(path):
            TESSERACT_CMD = path
            break

# Set Tesseract command if found
if TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    # Print will be done in main.py after encoding is set up
else:
    # Warning will be shown in main.py
    pass

