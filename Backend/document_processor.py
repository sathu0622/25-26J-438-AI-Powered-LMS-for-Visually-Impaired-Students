# document_processor.py
import os
import tempfile
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional 
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pdf2image
from pdf2image import convert_from_path

from config import IMG_SIZE, CLASS_NAMES, processed_documents
from ocr_processor import extract_text_book
from azure_processor import extract_with_azure
from summarizer import summarize_text, summarize_specific_article

def predict_resource_type(img_path: str, type_model) -> tuple[str, float]:
    """Predict resource type from image."""
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        pred = type_model.predict(arr, verbose=0)
        cls = CLASS_NAMES[np.argmax(pred)]
        conf = float(np.max(pred))
        return cls.lower(), conf
    except Exception as e:
        print(f"Resource type detection error: {e}")
        return "books", 0.5  # Default fallback

def extract_text_with_strategy(input_path: str, resource_type: str) -> Dict[str, Any]:
    print(f"\nExtracting text for resource type: {resource_type}")

    # ==================================================
    # 📰/📚 GEMINI EXTRACTION (BOOK / MAGAZINE / NEWSPAPER)
    # ==================================================
    if resource_type in ["newspapers", "magazine", "books"]:
        gemini_result = extract_with_azure(input_path, resource_type=resource_type)

        if gemini_result and gemini_result.get("articles"):
            print("✓ Gemini extraction successful")
            article_texts = [article.get("full_text", "") for article in gemini_result["articles"]]

            return {
                "full_text": gemini_result["full_text"],
                "article_texts": article_texts,
                "method": "gemini",
                "structured_articles": gemini_result["articles"],
            }

        print("⚠ Gemini extraction failed. Falling back to OCR")
        full_text = extract_text_book(input_path)
        return {
            "full_text": full_text,
            "article_texts": [full_text],
            "method": "simple_ocr_fallback",
            "structured_articles": [],
        }

    # Other resource labels fallback
    full_text = extract_text_book(input_path)
    return {
        "full_text": full_text,
        "article_texts": [full_text],
        "method": "simple_ocr",
        "structured_articles": [],
    }

def process_document(
    input_path: str, 
    type_model, 
    summ_tokenizer, 
    summ_model,
    processed_documents: Dict[str, Any]
) -> dict:
    """Process document through full pipeline."""
    try:
        print("\n" + "="*60)
        print("STARTING DOCUMENT PROCESSING")
        print("="*60)
        
        # Step 1: Detect resource type
        print("\n[1/4] Detecting resource type...")
        if input_path.lower().endswith(".pdf"):
            pages = convert_from_path(input_path, first_page=1, last_page=1)
            if pages:
                temp_dir = tempfile.gettempdir()
                temp_img = os.path.join(temp_dir, f"temp_detect_{os.urandom(4).hex()}.jpg")
                pages[0].save(temp_img, "JPEG", quality=95)
                resource_type, conf = predict_resource_type(temp_img, type_model)
                try:
                    os.remove(temp_img)
                except:
                    pass
            else:
                resource_type, conf = "books", 0.5
        else:
            resource_type, conf = predict_resource_type(input_path, type_model)
        
        print(f"✓ Resource type: {resource_type} (confidence: {conf:.2f})")
        
        # Step 2: Extract text with appropriate strategy
        print("\n[2/4] Extracting text with optimized strategy...")
        extraction_result = extract_text_with_strategy(input_path, resource_type)
        full_text = extraction_result["full_text"]
        article_texts = extraction_result["article_texts"]
        extraction_method = extraction_result["method"]
        structured_articles = extraction_result.get("structured_articles", [])
        
        if not full_text or len(full_text.strip()) == 0:
            raise ValueError("No text extracted from document")
        
        print(f"✓ Extracted {len(full_text)} characters using {extraction_method}")
        print(f"✓ Found {len(article_texts)} article(s)")
        
        # Step 3: Prepare structured article data if available
        article_list_for_selection = []
        if structured_articles:
            for i, article in enumerate(structured_articles[:10]):  # Limit to 10 articles for display
                article_list_for_selection.append({
                    "index": i + 1,
                    "article_id": article.get("article_id", f"article_{i+1}"),
                    "column": article.get("column", "unknown"),
                    "heading": article.get("heading", "No heading") or "No heading",
                    "subheading": article.get("subheading", "") or "",
                    "body_preview": " ".join(article.get("body", [])[:2])[:150] + "..." if article.get("body") else "",
                    "word_count": len(article.get("full_text", "").split()),
                    "paragraph_count": len(article.get("body", [])),
                    "is_main_article": i == 0  # First article is considered main
                })
        else:
            # Create dummy article for books or when no structure detected
            article_list_for_selection.append({
                "index": 1,
                "article_id": "full_document",
                "column": "full",
                "heading": "Full Document",
                "subheading": "Complete extracted text",
                "body_preview": full_text[:150] + "..." if len(full_text) > 150 else full_text,
                "word_count": len(full_text.split()),
                "paragraph_count": len(full_text.split('\n\n')),
                "is_main_article": True
            })
        
        # Step 4: Generate ONE summary for the main article only (not all articles)
        print("\n[3/4] Generating summary for main article...")
        summaries = []
        
        # Only summarize the first/main article initially
        if article_list_for_selection:
            main_article_id = article_list_for_selection[0]["article_id"]
            
            if structured_articles:
                # Find the main article in structured articles
                main_article_text = ""
                for article in structured_articles:
                    if article.get("article_id") == main_article_id:
                        main_article_text = article.get("full_text", "")
                        break
                
                if not main_article_text and structured_articles:
                    main_article_text = structured_articles[0].get("full_text", "")
            else:
                main_article_text = full_text
            
            if main_article_text.strip():
                print(f"  Summarizing main article: {article_list_for_selection[0]['heading'][:50]}...")
                summary = summarize_text(main_article_text, resource_type, summ_tokenizer, summ_model)
                summaries.append({
                    "article_id": main_article_id,
                    "heading": article_list_for_selection[0]["heading"],
                    "summary": summary,
                    "is_main": True
                })
            else:
                summaries.append({
                    "article_id": main_article_id,
                    "heading": article_list_for_selection[0]["heading"],
                    "summary": "Could not generate summary for this article.",
                    "is_main": True
                })
        
        print(f"✓ Generated {len(summaries)} summary for main article")
        
        # Create a document ID for later reference
        document_id = hashlib.md5(f"{input_path}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        
        # Store structured articles for later selective summarization AND Q&A
        processed_documents[document_id] = {
            "resource_type": resource_type,
            "structured_articles": structured_articles,
            "full_text": full_text,
            "timestamp": datetime.now().isoformat()
        }
        
        result = {
            "document_id": document_id,
            "resource_type": resource_type,
            "confidence": conf,
            "extracted_text_preview": full_text[:500] + "..." if len(full_text) > 500 else full_text,
            "article_list": article_list_for_selection,
            "summaries": summaries,  # Only main article summary
            "num_articles": len(article_list_for_selection),
            "text_length": len(full_text),
            "extraction_method": extraction_method,
            "processing_info": {
                "ocr_method": extraction_method,
                "article_detection": extraction_method == "gemini",
                "supports_selective_summarization": len(structured_articles) > 0,
                "supports_qa": True,  # NEW: All documents now support Q&A
                "timestamp": datetime.now().isoformat()
            }
        }
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print(f"Document ID: {document_id}")
        print("Use /summarize-article endpoint to get summaries for specific articles")
        print("Use /ask-question endpoint to ask questions about any article")  # NEW
        print("="*60 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Processing error: {str(e)}")