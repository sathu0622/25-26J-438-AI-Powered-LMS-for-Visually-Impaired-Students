# Document Processor Backend

FastAPI backend for document processing with OCR, grammar correction, resource type detection, and summarization.

## Features

- **OCR Extraction**: Extract text from PDFs and images using Tesseract
- **Grammar Correction**: AI-powered grammar and OCR error correction
- **Resource Type Detection**: Automatically detect if document is a Book, Magazine, or Newspaper
- **Smart Summarization**: Type-specific summarization using fine-tuned T5 model

## Prerequisites

### Windows Setup

1. **Install Tesseract OCR**:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location: `C:\Program Files\Tesseract-OCR`
   - Or update the path in `main.py` if installed elsewhere

2. **Poppler (for PDF processing)**:
   - Already included in `Release-25.12.0-0/poppler-25.12.0/`
   - The application will automatically use it

3. **Python 3.12.12**:
   - Ensure Python 3.12.12 is installed

### Linux/Mac Setup

1. **Install Tesseract**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr
   
   # Mac
   brew install tesseract
   ```

2. **Install Poppler**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y poppler-utils
   
   # Mac
   brew install poppler
   ```

## Installation

1. **Navigate to Backend directory**:
   ```bash
   cd Backend
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /
```
Returns API status and model loading information.

### Process Document
```
POST /process
Content-Type: multipart/form-data
Body: file (PDF or image file)
```

**Response**:
```json
{
  "resource_type": "books",
  "confidence": 0.95,
  "extracted_text": "Full extracted and corrected text...",
  "summaries": ["Summary 1", "Summary 2"],
  "num_articles": 1
}
```

## Model Files

The following model files should be present in the `Model/` directory:

- `book_magazine_newspaper_model_super_finetuned2.keras` - Resource type detection model
- `final/` - T5 summarization model with PEFT adapter
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `tokenizer_config.json`
  - `spiece.model`
  - Other tokenizer files

## Troubleshooting

### Tesseract not found
- Windows: Ensure Tesseract is installed and update the path in `main.py`
- Linux/Mac: Install Tesseract using package manager

### Poppler not found
- Windows: Ensure `Release-25.12.0-0/poppler-25.12.0/Library/bin` exists
- Linux/Mac: Install poppler-utils

### Model loading errors
- Ensure all model files are in the correct directories
- Check that you have sufficient RAM/VRAM
- For GPU support, ensure CUDA is properly installed

### Memory issues
- The models are loaded into memory at startup
- Ensure you have at least 8GB RAM available
- For large documents, processing may take time

## Development

The frontend expects the API to be running on `http://localhost:8000` by default. You can change this in the frontend's `.env` file or `FileUpload.jsx`.

## Notes

- First request may be slower as models are loaded
- Large PDFs may take several minutes to process
- GPU is recommended but not required (CPU will work, just slower)

