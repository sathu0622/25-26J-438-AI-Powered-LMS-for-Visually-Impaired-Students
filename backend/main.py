from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import models_loader
from schemas import EvaluationRequest, EvaluationResponse
from evaluation import evaluate_student_answer
from logger_config import logger
from braille_decoder import decode_pdf

import torch
import tempfile
import re


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


@app.post("/decode")
async def decode_braille(file: UploadFile = File(...)):
    """
    Upload a scanned Braille PDF.
    Returns JSON with 'question' (page 1) and 'answer' (pages 2+).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await file.read()
        all_text = decode_pdf(pdf_bytes)
    except Exception as e:
        logger.error(f"Braille decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    if not all_text:
        raise HTTPException(status_code=400, detail="PDF has no pages.")

    def clean_text(text: str) -> str:
        """Remove newlines and fix mid-word splits from Braille line wrapping."""
        text = re.sub(r'(\w)\n(\w)', r'\1\2', text)  # "o\nf" → "of"
        text = re.sub(r'\n', ' ', text)               # remaining \n → space
        text = re.sub(r' +', ' ', text)               # collapse multiple spaces
        return text.strip()

    # Page 1 = Question, Pages 2+ = Answer
    question = clean_text(all_text[0])
    answer   = clean_text(" ".join(all_text[1:]))
    full_text = f"Question: {question}\n\nAnswer: {answer}"

    return JSONResponse(content={
        "status": "success",
        "question": question,
        "answer": answer,
        "full_text": full_text
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)