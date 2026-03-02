# router.py
# FastAPI route definitions

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from braille_decoder import decode_pdf

router = APIRouter()


@router.post("/decode")
async def decode_braille(file: UploadFile = File(...)):
    """
    Upload a scanned Braille PDF.
    Returns JSON with 'question' (page 1) and 'answer' (pages 2+).
    """
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported."}
        )

    try:
        pdf_bytes = await file.read()
        all_text = decode_pdf(pdf_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process PDF: {str(e)}"}
        )

    if not all_text:
        return JSONResponse(
            status_code=400,
            content={"error": "PDF has no pages."}
        )

    # Page 1 = Question, Pages 2+ = Answer
    question = all_text[0].strip()
    answer   = " ".join(all_text[1:]).strip()

    return JSONResponse(content={
        "filename": file.filename,
        "total_pages": len(all_text),
        "question": question,
        "answer": answer
    })