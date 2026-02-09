# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import models_loader
from schemas import EvaluationRequest, EvaluationResponse
from evaluation import evaluate_student_answer
from logger_config import logger
import torch
import numpy as np
import cv2
import tempfile
from braille_decoder import braille_image_to_text  # Correct function name

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading models...")
    models_loader.load_models()
    yield
    logger.info("Application shutdown: Clearing cache...")
    if models_loader.model and torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="AI-Powered LMS for Visually Impaired Students",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Existing Endpoints
# ========================

@app.get("/")
async def root():
    return {
        "message": "AI-Powered LMS Backend",
        "services": ["Answer Evaluation", "Braille OCR"],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": models_loader.model is not None
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(request: EvaluationRequest):
    if not request.question or not request.student_answer:
        raise HTTPException(status_code=400, detail="Question and student_answer cannot be empty")

    if not models_loader.model or not models_loader.sbert:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result = evaluate_student_answer(
        request.question,
        request.student_answer,
        models_loader.tokenizer,
        models_loader.model,
        models_loader.sbert
    )
    return EvaluationResponse(**result)

# ========================
# Braille OCR Endpoint
# ========================

@app.post("/braille/convert")
async def convert_braille(image: UploadFile = File(...)):
    tmp_path = None  # initialize

    try:
        # Save uploaded file to a temporary file
        suffix = ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        # Convert Braille image → English text
        text = braille_image_to_text(tmp_path, debug=False)
    finally:
        # Remove temp file safely
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"text": text}


# ========================
# Run Server (PORT 8080)
# ========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
