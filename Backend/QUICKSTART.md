# Quick Start Guide

## Windows Setup (5 minutes)

### Step 1: Install Tesseract OCR
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Install to default location: `C:\Program Files\Tesseract-OCR`
4. ✅ Done! (Poppler is already included in the project)

### Step 2: Setup Python Environment
```bash
cd Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the Server
```bash
python main.py
```

Or simply double-click `start.bat`

### Step 4: Start Frontend
In a new terminal:
```bash
cd Frontend
npm install
npm run dev
```

## Linux/Mac Setup

### Step 1: Install Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Mac
brew install tesseract poppler
```

### Step 2: Setup Python Environment
```bash
cd Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run the Server
```bash
python main.py
```

Or use the script:
```bash
chmod +x start.sh
./start.sh
```

## Testing

1. Open browser to `http://localhost:5173` (frontend)
2. Upload a PDF or image file
3. Wait for processing (may take 1-5 minutes depending on file size)
4. View results!

## Troubleshooting

### "Tesseract not found"
- **Windows**: Reinstall Tesseract or update path in `main.py` line 30-35
- **Linux/Mac**: Run `sudo apt-get install tesseract-ocr` or `brew install tesseract`

### "Model loading failed"
- Check that all model files exist in `Backend/Model/`
- Ensure you have enough RAM (8GB+ recommended)
- First run will download models from HuggingFace (requires internet)

### "Port already in use"
- Change port in `main.py` last line: `uvicorn.run(app, host="0.0.0.0", port=8001)`
- Update frontend `.env` or `FileUpload.jsx` with new port

### Slow processing
- Normal for large files
- GPU will speed up (if available)
- First request is slower (model initialization)

## API Testing

Test the API directly:
```bash
curl -X POST "http://localhost:8000/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

