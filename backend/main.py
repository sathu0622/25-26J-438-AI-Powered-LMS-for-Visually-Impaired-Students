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
import tempfile

# ✅ Import Braille PDF Decoder
from braille_decoder import decode_braille_pdf


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
# Root Endpoints
# ========================

@app.get("/")
async def root():
    return {
        "message": "AI-Powered LMS Backend",
        "services": ["Answer Evaluation", "Braille PDF OCR"],
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": models_loader.model is not None
    }


# ========================
# Answer Evaluation Endpoint
# ========================

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
# ✅ Braille Unicode PDF Endpoint
# ========================

@app.post("/braille/convert-pdf")
async def convert_braille_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(await file.read())
            temp_path = temp_pdf.name

        # Decode Braille PDF
        result = decode_braille_pdf(temp_path)

        # Remove temp file
        os.remove(temp_path)

        return {
            "status": "success",
            "question": result["question"],
            "answer": result["answer"],
            "full_text": result["full_text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Braille PDF decoding failed: {str(e)}")


# ========================
# Run Server
# ========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
