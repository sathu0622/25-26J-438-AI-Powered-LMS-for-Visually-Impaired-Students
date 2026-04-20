# api_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import tempfile
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import processed_documents
from models import load_all_models
from document_processor import process_document
from summarizer import summarize_specific_article, summarize_text
from qa_system import answer_question_for_article
from favorite_articles import get_favorite_store
from syllabus_matcher import SyllabusMatcher

# Pydantic Models
class QARequest(BaseModel):
    document_id: str = Field(..., description="Document ID from processing")
    article_id: str = Field(..., description="Article ID to ask questions about")
    question: str = Field(..., min_length=1, description="Question to ask")
    max_answer_len: int = Field(128, ge=1, le=256, description="Maximum answer token length")
    score_threshold: float = Field(0.08, ge=0.0, le=1.0, description="Confidence threshold")

class QAResponse(BaseModel):
    document_id: str
    article_id: str
    question: str
    answer: str
    confidence: float
    context_preview: Optional[str] = None
    resource_type: Optional[str] = None
    timestamp: str

class FavoriteArticleRequest(BaseModel):
    document_id: str = Field(..., description="Document ID from processing")
    article_id: str = Field(..., description="Article ID to save as favorite")
    summary: Optional[str] = Field(
        None,
        description="Optional summary text to persist (e.g. current UI summary). "
        "If omitted, the server uses cached summary from processing/summarize-article when available.",
    )


def _article_full_text(article: Dict[str, Any]) -> str:
    text = (article.get("full_text") or "").strip()
    if text:
        return text
    parts: List[str] = []
    if article.get("heading"):
        parts.append(str(article["heading"]))
    if article.get("subheading"):
        parts.append(str(article["subheading"]))
    body = article.get("body")
    if body:
        parts.append("\n".join(filter(None, body)))
    return "\n".join(parts).strip()


def _summary_from_document_cache(document_data: Dict[str, Any], article_id: str) -> str:
    for entry in document_data.get("summaries") or []:
        if entry.get("article_id") == article_id:
            return (entry.get("summary") or "").strip()
    return ""


def _upsert_article_summary_cache(
    document_id: str,
    article_id: str,
    summary: str,
    heading: str = "",
    subheading: str = "",
) -> None:
    doc = processed_documents.get(document_id)
    if not doc:
        return
    summaries = doc.setdefault("summaries", [])
    for item in summaries:
        if item.get("article_id") == article_id:
            item["summary"] = summary
            if heading:
                item["heading"] = heading
            if subheading is not None:
                item["subheading"] = subheading
            return
    summaries.append(
        {
            "article_id": article_id,
            "heading": heading or "No heading",
            "subheading": subheading or "",
            "summary": summary,
            "is_main": False,
        }
    )

class FavoriteArticleDeleteRequest(BaseModel):
    document_id: str = Field(..., description="Document ID of favorite article")
    article_id: str = Field(..., description="Article ID of favorite article")


class SyllabusMatchRequest(BaseModel):
    document_id: str = Field(..., description="Document ID from processing")
    article_id: str = Field(..., description="Article ID to classify against syllabus")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence to mark as in syllabus")

# Load models globally
models = load_all_models()
favorite_store = None
favorite_store_error = None
try:
    favorite_store = get_favorite_store()
except Exception as exc:
    favorite_store_error = str(exc)

syllabus_matcher = SyllabusMatcher(
    Path(__file__).parent / "syllabus" / "Gr 10 11.xlsx"
)

app = FastAPI(title="Article Detection Document Processor with Q&A", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "EXACT Colab Article Detection Document Processor API v6.0 with Q&A",
        "features": {
            "simple_ocr": True,
            "azure_document_intelligence": True,
            "resource_type_detection": True,
            "summarization": True,
            "exact_colab_article_detection": True,
            "column_based_article_grouping": True,
            "selective_article_summarization": True,
            "question_answering": True,
            "favorite_articles": True
        },
        "models_loaded": {
            "resource_type": models["type_model"] is not None,
            "summarization": models["summ_model"] is not None,
            "question_answering": models["qa_model"] is not None
        },
        "qa_model": "distilbert-base-cased-distilled-squad",
        "azure_configured": True,
        "endpoints": {
            "POST /process": "Process a document and extract articles",
            "POST /summarize-article": "Summarize a specific article by ID or heading",
            "POST /ask-question": "Ask questions about a specific article",
            "POST /syllabus-match": "Identify if an article belongs to syllabus chapter/topic",
            "POST /favorites": "Add an article to favorites (shared for all users)",
            "GET /favorites": "List all favorite articles (shared for all users)",
            "DELETE /favorites": "Remove an article from favorites",
            "GET /articles/{document_id}": "Get article list for a processed document",
            "GET /cleanup": "Clean up old documents"
        }
    }

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Process uploaded document."""
    allowed_types = [
        "application/pdf",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/tiff",
        "image/bmp"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    temp_dir = tempfile.gettempdir()
    file_ext = os.path.splitext(file.filename)[1] or (".pdf" if file.content_type == "application/pdf" else ".jpg")
    temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{file_ext}")
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\n{'='*60}")
        print(f"Processing uploaded file: {file.filename}")
        print(f"{'='*60}")
        
        result = process_document(
            temp_path, 
            models["type_model"], 
            models["summ_tokenizer"], 
            models["summ_model"],
            processed_documents
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.post("/summarize-article")
async def summarize_specific_article_endpoint(
    document_id: str = Form(...),
    article_id: str = Form(...)
):
    """Summarize a specific article from a previously processed document."""
    
    if document_id not in processed_documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document ID {document_id} not found. Please process the document first."
        )
    
    document_data = processed_documents[document_id]
    resource_type = document_data["resource_type"]
    structured_articles = document_data.get("structured_articles", [])
    full_text = document_data["full_text"]
    
    print(f"\n{'='*60}")
    print(f"Summarizing article {article_id} from document {document_id}")
    print(f"{'='*60}")
    
    if not structured_articles:
        # For books or documents without structured articles
        if article_id == "full_document":
            # Summarize the full document
            summary = summarize_text(full_text, resource_type, models["summ_tokenizer"], models["summ_model"])
            _upsert_article_summary_cache(
                document_id,
                article_id,
                summary,
                "Full Document",
                "Complete extracted text",
            )
            return {
                "document_id": document_id,
                "article_id": article_id,
                "heading": "Full Document",
                "summary": summary,
                "word_count": len(full_text.split()),
                "full_text_preview": full_text[:300] + "..." if len(full_text) > 300 else full_text,
                "resource_type": resource_type,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="This document does not have structured articles. Use 'full_document' as article_id."
            )
    
    # Find and summarize the specific article
    result = summarize_specific_article(
        structured_articles, 
        article_id, 
        resource_type,
        models["summ_tokenizer"],
        models["summ_model"]
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=404,
            detail=f"Article '{article_id}' not found. Available articles: {result.get('available_articles', [])}"
        )
    
    # Add document info
    result["document_id"] = document_id
    result["resource_type"] = resource_type
    _upsert_article_summary_cache(
        document_id,
        result.get("article_id", article_id),
        result.get("summary", ""),
        result.get("heading", "") or "",
        result.get("subheading", "") or "",
    )

    return JSONResponse(content=result)

@app.post("/ask-question")
async def ask_question_endpoint(
    request: QARequest
):
    """Ask a question about a specific article with better error handling."""
    
    print(f"\n{'='*60}")
    print(f"Question about document {request.document_id}, article {request.article_id}")
    print(f"Question: {request.question}")
    print(f"Max answer length: {request.max_answer_len}")
    print(f"Score threshold: {request.score_threshold}")
    print(f"{'='*60}")
    
    try:
        result = answer_question_for_article(
            document_id=request.document_id,
            article_id=request.article_id,
            question=request.question,
            processed_documents=processed_documents,
            qa_tokenizer=models["qa_tokenizer"],
            qa_model=models["qa_model"],
            max_answer_len=request.max_answer_len,
            score_threshold=request.score_threshold
        )
        
        if "error" in result:
            status_code = 404 if "not found" in result["error"].lower() else 400
            raise HTTPException(
                status_code=status_code,
                detail=result["error"]
            )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in Q&A: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)[:100]}"
        )


@app.post("/syllabus-match")
async def syllabus_match_endpoint(request: SyllabusMatchRequest):
    """Match one extracted article against chapter-wise syllabus from Excel."""
    if request.document_id not in processed_documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document ID {request.document_id} not found."
        )

    if syllabus_matcher.error:
        raise HTTPException(
            status_code=500,
            detail=f"Syllabus matcher unavailable: {syllabus_matcher.error}"
        )

    document_data = processed_documents[request.document_id]
    structured_articles = document_data.get("structured_articles", [])
    full_text = document_data.get("full_text", "")
    target_article: Dict[str, Any]

    if structured_articles:
        target_article = next(
            (a for a in structured_articles if a.get("article_id") == request.article_id),
            {}
        )
        if not target_article:
            raise HTTPException(
                status_code=404,
                detail=f"Article '{request.article_id}' not found in document '{request.document_id}'."
            )
        article_text = _article_full_text(target_article)
        heading = target_article.get("heading", "") or ""
    else:
        if request.article_id != "full_document":
            raise HTTPException(
                status_code=400,
                detail="Only 'full_document' is valid for this document."
            )
        article_text = full_text
        heading = "Full Document"

    match_result = syllabus_matcher.match_article(
        article_text=article_text,
        article_heading=heading,
        threshold=request.threshold,
    )
    if "error" in match_result:
        raise HTTPException(status_code=400, detail=match_result["error"])

    confidence = float(match_result.get("confidence", 0.0) or 0.0)
    # Enforce a backend minimum so low-confidence matches are never exposed as in-syllabus.
    effective_threshold = max(request.threshold, 0.5)
    is_under_subject = confidence > effective_threshold

    if is_under_subject:
        response_result = {
            "in_syllabus": True,
            "confidence": confidence,
            "match": match_result.get("match"),
            "method": match_result.get("method", "unknown"),
            "message": "Content is under this subject.",
        }
    else:
        response_result = {
            "in_syllabus": False,
            "confidence": confidence,
            "match": None,
            "method": match_result.get("method", "unknown"),
            "message": "Content is not under the subject.",
        }

    return {
        "document_id": request.document_id,
        "article_id": request.article_id,
        "resource_type": document_data.get("resource_type", ""),
        "result": response_result,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/articles/{document_id}")
async def get_articles_list(document_id: str):
    """Get the list of articles for a previously processed document."""
    
    if document_id not in processed_documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document ID {document_id} not found."
        )
    
    document_data = processed_documents[document_id]
    structured_articles = document_data.get("structured_articles", [])
    
    article_list = []
    if structured_articles:
        for i, article in enumerate(structured_articles[:20]):  # Limit to 20 articles
            article_list.append({
                "index": i + 1,
                "article_id": article.get("article_id", f"article_{i+1}"),
                "column": article.get("column", "unknown"),
                "heading": article.get("heading", "No heading") or "No heading",
                "subheading": article.get("subheading", "") or "",
                "body_preview": " ".join(article.get("body", [])[:2])[:150] + "..." if article.get("body") else "",
                "word_count": len(article.get("full_text", "").split()),
                "paragraph_count": len(article.get("body", []))
            })
    else:
        # For books or documents without structured articles
        article_list.append({
            "index": 1,
            "article_id": "full_document",
            "column": "full",
            "heading": "Full Document",
            "subheading": "Complete extracted text",
            "body_preview": document_data["full_text"][:150] + "..." if len(document_data["full_text"]) > 150 else document_data["full_text"],
            "word_count": len(document_data["full_text"].split()),
            "paragraph_count": len(document_data["full_text"].split('\n\n'))
        })
    
    return {
        "document_id": document_id,
        "resource_type": document_data["resource_type"],
        "num_articles": len(article_list),
        "articles": article_list,
        "timestamp": document_data["timestamp"],
        "supports_qa": True  # All documents support Q&A
    }

@app.post("/favorites")
async def add_favorite_article(request: FavoriteArticleRequest):
    """Add an article to globally shared favorites."""
    if favorite_store is None:
        raise HTTPException(
            status_code=503,
            detail=f"Favorites storage unavailable: {favorite_store_error}"
        )

    if request.document_id not in processed_documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document ID {request.document_id} not found."
        )

    document_data = processed_documents[request.document_id]
    structured_articles = document_data.get("structured_articles", [])
    resource_type = document_data.get("resource_type", "")

    article_to_store: Dict[str, Any] = {
        "document_id": request.document_id,
        "article_id": request.article_id,
        "resource_type": resource_type,
    }

    if not structured_articles:
        if request.article_id != "full_document":
            raise HTTPException(
                status_code=400,
                detail="Only 'full_document' can be favorited for this document."
            )
        full_text = document_data.get("full_text", "")
        body_preview = full_text[:150] + "..." if len(full_text) > 150 else full_text
        article_to_store.update(
            {
                "heading": "Full Document",
                "subheading": "Complete extracted text",
                "body_preview": body_preview,
                "full_content": full_text,
            }
        )
    else:
        target_article = None
        for article in structured_articles:
            if article.get("article_id") == request.article_id:
                target_article = article
                break

        if not target_article:
            raise HTTPException(
                status_code=404,
                detail=f"Article '{request.article_id}' not found in document '{request.document_id}'."
            )

        full_content = _article_full_text(target_article)
        body_preview = " ".join(target_article.get("body", [])[:2])[:150]
        if body_preview:
            body_preview += "..."

        article_to_store.update(
            {
                "heading": target_article.get("heading", "No heading") or "No heading",
                "subheading": target_article.get("subheading", "") or "",
                "body_preview": body_preview,
                "full_content": full_content,
            }
        )

    summary_text = (request.summary or "").strip()
    if not summary_text:
        summary_text = _summary_from_document_cache(document_data, request.article_id)
    if not summary_text and models.get("summ_tokenizer") and models.get("summ_model"):
        if not structured_articles and request.article_id == "full_document":
            ft = document_data.get("full_text", "")
            if ft.strip():
                summary_text = summarize_text(
                    ft, resource_type, models["summ_tokenizer"], models["summ_model"]
                )
                _upsert_article_summary_cache(
                    request.document_id,
                    request.article_id,
                    summary_text,
                    "Full Document",
                    "Complete extracted text",
                )
        elif structured_articles:
            summ_result = summarize_specific_article(
                structured_articles,
                request.article_id,
                resource_type,
                models["summ_tokenizer"],
                models["summ_model"],
            )
            if "error" not in summ_result:
                summary_text = summ_result.get("summary") or ""
                _upsert_article_summary_cache(
                    request.document_id,
                    request.article_id,
                    summary_text,
                    article_to_store.get("heading", "") or "",
                    article_to_store.get("subheading", "") or "",
                )
    article_to_store["summary"] = summary_text

    try:
        saved = favorite_store.add_favorite(article_to_store)
        return {
            "message": "Article added to favorites",
            "favorite": saved
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save favorite: {exc}")

@app.get("/favorites")
async def list_favorite_articles():
    """List all globally shared favorite articles."""
    if favorite_store is None:
        raise HTTPException(
            status_code=503,
            detail=f"Favorites storage unavailable: {favorite_store_error}"
        )

    try:
        favorites = favorite_store.list_favorites()
        return {
            "count": len(favorites),
            "favorites": favorites
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load favorites: {exc}")

@app.delete("/favorites")
async def remove_favorite_article(request: FavoriteArticleDeleteRequest):
    """Remove a favorite article from globally shared favorites."""
    if favorite_store is None:
        raise HTTPException(
            status_code=503,
            detail=f"Favorites storage unavailable: {favorite_store_error}"
        )

    try:
        deleted_count = favorite_store.remove_favorite(
            document_id=request.document_id,
            article_id=request.article_id
        )
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Favorite article not found.")
        return {"message": "Favorite article removed"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove favorite: {exc}")

@app.get("/cleanup")
async def cleanup_old_documents(hours: int = 24):
    """Clean up old processed documents."""
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    removed_count = 0
    
    for doc_id in list(processed_documents.keys()):
        try:
            doc_time = datetime.fromisoformat(processed_documents[doc_id]["timestamp"]).timestamp()
            if doc_time < cutoff_time:
                del processed_documents[doc_id]
                removed_count += 1
        except:
            # If timestamp parsing fails, remove the document
            del processed_documents[doc_id]
            removed_count += 1
    
    return {
        "message": f"Cleaned up {removed_count} old documents",
        "remaining_documents": len(processed_documents),
        "cleanup_hours": hours
    }

def check_and_install_azure_packages():
    """Check if Azure packages are installed, install if not."""
    try:
        import azure.ai.formrecognizer
        import azure.core
        print("✓ Azure packages already installed")
    except ImportError:
        print("⚠ Azure packages not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "azure-ai-formrecognizer", "azure-core"])
        print("✓ Azure packages installed")