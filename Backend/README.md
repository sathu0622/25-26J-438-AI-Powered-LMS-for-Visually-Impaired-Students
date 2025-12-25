# Document Processing Backend

FastAPI backend for processing PDFs, books, magazines, and newspapers with OCR, resource type detection, and summarization.

## Setup Instructions

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Windows:**
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. Add to PATH or set environment variable:
   ```powershell
   $env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### 3. Install Poppler (for PDF to image conversion)

**Windows:**
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract and add `bin` folder to PATH

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**Mac:**
```bash
brew install poppler
```

### 4. Configure Environment Variables

Create a `.env` file in the Backend directory (optional, if Tesseract is not in PATH):

```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 5. Download Models (if not already present)

The models should be in the `Model/` directory:
- `book_magazine_newspaper_model_super_finetuned2.keras` - Resource type detection
- `final/` - LoRA adapter for T5 summarization model

The base models will be downloaded automatically on first run:
- `google/flan-t5-base` - Base T5 model
- `prithivida/grammar_error_correcter_v1` - Grammar correction model

### 6. Run the Server

```bash
# Make sure virtual environment is activated
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 7. API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check with model status
- `POST /process` - Upload and process a document (PDF or image)

### API Usage Example

```python
import requests

files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/process', files=files)
result = response.json()
```

## Model Information

- **Resource Type Detection**: Keras model trained to classify Books, Magazines, and Newspapers
- **Summarization**: FLAN-T5-base with LoRA adapter fine-tuned for summarization
- **Grammar Correction**: prithivida/grammar_error_correcter_v1

## Notes

- First run will download base models (may take several minutes)
- Processing time depends on document size and complexity
- GPU recommended for faster processing (CUDA compatible)





