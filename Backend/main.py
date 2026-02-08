# """
# Fixed Article Detection for Document Processing
# With EXACT Colab logic for article detection
# Added: Selective article summarization
# Added: Question Answering system
# """

# import os
# import sys
# import cv2
# import pytesseract
# import pdf2image
# from pdf2image import convert_from_path
# from PIL import Image
# import json
# import numpy as np
# import re
# import tempfile
# from pathlib import Path
# from typing import Optional, List, Tuple, Dict, Any
# from datetime import datetime

# import torch
# from transformers import (
#     T5Tokenizer, 
#     T5ForConditionalGeneration,
#     AutoTokenizer,
#     AutoModelForQuestionAnswering
# )
# from peft import PeftModel
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field

# # ============================================================
# # CONFIGURATION
# # ============================================================

# BASE_DIR = Path(__file__).parent
# MODEL_DIR = BASE_DIR / "Model"

# # Tesseract configuration
# if sys.platform == "win32":
#     tesseract_paths = [
#         r"C:\Program Files\Tesseract-OCR\tesseract.exe",
#         r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
#     ]
#     for path in tesseract_paths:
#         if os.path.exists(path):
#             pytesseract.pytesseract.tesseract_cmd = path
#             break
#     else:
#         print("Warning: Tesseract not found. Please install Tesseract-OCR")
    
#     poppler_path = BASE_DIR.parent / "Release-25.12.0-0" / "poppler-25.12.0" / "Library" / "bin"
#     if poppler_path.exists():
#         os.environ["PATH"] = str(poppler_path) + os.pathsep + os.environ.get("PATH", "")
# else:
#     # Linux/Mac configuration
#     pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# # Model paths
# TYPE_MODEL_PATH = MODEL_DIR / "book_magazine_newspaper_model_super_finetuned_FIXED.keras"
# T5_MODEL_DIR = MODEL_DIR / "final"

# # Configuration
# IMG_SIZE = (224, 224)
# CLASS_NAMES = ['Books', 'Magazine', 'Newspapers']
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Azure Document Intelligence Configuration

# # Q&A Model Configuration
# QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# # ============================================================
# # MODEL LOADING
# # ============================================================

# print("Loading models...")

# type_model = load_model(str(TYPE_MODEL_PATH), compile=False)
# print(f"✓ Resource type model loaded")

# base_model_name = "google/flan-t5-base"
# summ_tokenizer = T5Tokenizer.from_pretrained(base_model_name)
# base_summ_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
# summ_model = PeftModel.from_pretrained(base_summ_model, str(T5_MODEL_DIR))
# summ_model.to(DEVICE)
# summ_model.eval()
# print(f"✓ T5 summarization model loaded ({DEVICE})")

# # Load Q&A model
# qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
# qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
# qa_model.to(DEVICE)
# qa_model.eval()
# print(f"✓ Q&A model loaded ({DEVICE})")

# print("Model loading complete!\n")

# # ============================================================
# # PYDANTIC MODELS
# # ============================================================

# class QARequest(BaseModel):
#     document_id: str = Field(..., description="Document ID from processing")
#     article_id: str = Field(..., description="Article ID to ask questions about")
#     question: str = Field(..., min_length=1, description="Question to ask")
#     max_answer_len: int = Field(64, ge=1, le=256, description="Maximum answer token length")
#     score_threshold: float = Field(0.15, ge=0.0, le=1.0, description="Confidence threshold")

# class QAResponse(BaseModel):
#     document_id: str
#     article_id: str
#     question: str
#     answer: str
#     confidence: float
#     context_preview: Optional[str] = None
#     resource_type: Optional[str] = None
#     timestamp: str

# # ============================================================
# # SIMPLIFIED OCR FUNCTIONS (For Books)
# # ============================================================

# def ocr_image_simple(image_path: str) -> str:
#     """
#     Simple OCR function using basic Tesseract.
#     """
#     try:
#         print(f"Processing image: {image_path}")
        
#         # Load image using OpenCV
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"  Warning: Could not read image with OpenCV, trying PIL fallback...")
#             # Fallback to PIL
#             pil_img = Image.open(image_path)
#             # Convert PIL image to numpy array
#             img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
#         height, width = img.shape[:2]
#         print(f"  Image size: {width}x{height}")
        
#         # Convert to RGB (Tesseract needs RGB)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Run OCR with standard configuration
#         text = pytesseract.image_to_string(img_rgb, lang="eng")
        
#         print(f"  Extracted {len(text)} characters")
#         return text.strip()
        
#     except Exception as e:
#         print(f"OCR Error: {e}")
#         # Try alternative method
#         try:
#             # Direct PIL approach
#             pil_img = Image.open(image_path)
#             text = pytesseract.image_to_string(pil_img, lang="eng")
#             return text.strip()
#         except Exception as e2:
#             print(f"Fallback OCR also failed: {e2}")
#             return ""

# def ocr_pdf_simple(pdf_path: str) -> str:
#     """Extract text from PDF using simple OCR on all pages."""
#     try:
#         print(f"Processing PDF: {pdf_path}")
        
#         # Convert PDF to images with standard DPI
#         print("  Converting PDF to images...")
#         pages = convert_from_path(pdf_path, dpi=200)
        
#         print(f"  Converted {len(pages)} pages")
        
#         all_pages_text = []
        
#         for page_num, page in enumerate(pages):
#             print(f"  Processing page {page_num + 1}/{len(pages)}...")
            
#             # Save page as temporary image
#             temp_dir = tempfile.gettempdir()
#             temp_img = os.path.join(temp_dir, f"temp_page_{page_num}_{os.urandom(4).hex()}.png")
            
#             page.save(temp_img, "PNG", dpi=(200, 200))
            
#             # Process with simple OCR
#             page_text = ocr_image_simple(temp_img)
            
#             if page_text.strip():
#                 all_pages_text.append(page_text)
            
#             # Clean up
#             try:
#                 os.remove(temp_img)
#             except:
#                 pass
        
#         # Combine all pages with page break markers
#         combined = "\n\n[PAGE BREAK]\n\n".join(all_pages_text)
#         print(f"  Total extracted characters: {len(combined)}")
        
#         return combined
        
#     except Exception as e:
#         print(f"PDF OCR Error: {e}")
#         # Try to get at least some text
#         try:
#             # Try with first page only
#             pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
#             if pages:
#                 temp_dir = tempfile.gettempdir()
#                 temp_img = os.path.join(temp_dir, f"temp_fallback_{os.urandom(4).hex()}.png")
#                 pages[0].save(temp_img, "PNG", dpi=(200, 200))
#                 text = ocr_image_simple(temp_img)
#                 try:
#                     os.remove(temp_img)
#                 except:
#                     pass
#                 return text
#         except:
#             pass
#         return ""

# def extract_text_book(input_path: str) -> str:
#     """Extract text for books using simple OCR."""
#     if input_path.lower().endswith(".pdf"):
#         return ocr_pdf_simple(input_path)
#     else:
#         return ocr_image_simple(input_path)

# # ============================================================
# # AZURE DOCUMENT INTELLIGENCE FUNCTIONS (EXACT COLAB LOGIC)
# # ============================================================

# def extract_with_azure(file_path: str) -> Dict[str, Any]:
#     """
#     Extract text and layout using Azure Document Intelligence.
#     This is specifically for newspapers and magazines.
#     """
#     try:
#         from azure.ai.formrecognizer import DocumentAnalysisClient
#         from azure.core.credentials import AzureKeyCredential
#         from azure.core.exceptions import HttpResponseError
        
#         print(f"Using Azure Document Intelligence for enhanced extraction...")
        
#         # Initialize Azure client
#         client = DocumentAnalysisClient(
#             endpoint=AZURE_ENDPOINT,
#             credential=AzureKeyCredential(AZURE_KEY)
#         )
        
#         # Open and analyze the file
#         with open(file_path, "rb") as f:
#             poller = client.begin_analyze_document(
#                 model_id="prebuilt-layout",
#                 document=f
#             )
        
#         result = poller.result()
        
#         # Extract all text
#         full_text = ""
#         if hasattr(result, 'content'):
#             full_text = result.content
#         else:
#             # Fallback: combine all paragraphs
#             paragraphs = []
#             for p in result.paragraphs:
#                 paragraphs.append(p.content)
#             full_text = "\n".join(paragraphs)
        
#         # Extract structured articles using EXACT Colab logic
#         articles = extract_articles_exact_colab_logic(result)
        
#         return {
#             "full_text": full_text,
#             "articles": articles,
#             "raw_result": result,
#             "method": "azure_document_intelligence"
#         }
        
#     except ImportError:
#         print("Azure packages not installed. Falling back to simple OCR...")
#         return None
#     except HttpResponseError as e:
#         print(f"Azure API error: {e}. Falling back to simple OCR...")
#         return None
#     except Exception as e:
#         print(f"Azure processing error: {e}. Falling back to simple OCR...")
#         return None

# def extract_articles_exact_colab_logic(result) -> List[Dict[str, Any]]:
#     """
#     EXACT COLAB LOGIC: Extract structured articles from Azure Document Intelligence result.
#     This is your exact Colab code converted to Python.
#     """
#     try:
#         # Step 1: Group paragraphs by column (left/right) EXACTLY like Colab
#         articles_by_column = {}
        
#         for p in result.paragraphs:
#             if not p.bounding_regions:
#                 continue
                
#             box = p.bounding_regions[0].polygon
            
#             # Calculate average X position (EXACT Colab logic)
#             x_avg = sum(point.x for point in box) / len(box)
            
#             # Determine column (EXACT Colab logic)
#             column = "left" if x_avg < 0.5 else "right"
            
#             if column not in articles_by_column:
#                 articles_by_column[column] = []
            
#             # Add paragraph with its original properties
#             articles_by_column[column].append(p)
        
#         print(f"Paragraphs grouped by column: { {k: len(v) for k, v in articles_by_column.items()} }")
        
#         # Step 2: Function to split paragraphs into articles (EXACT Colab function)
#         def split_into_articles(paragraphs):
#             article_list = []
#             current_article = None

#             for p in paragraphs:
#                 text = p.content.strip()
#                 role = getattr(p, "role", "")  # use role if exists

#                 if len(text) < 5:
#                     continue

#                 if role == "title":
#                     if current_article:
#                         article_list.append(current_article)
#                     current_article = {
#                         "heading": text,
#                         "subheading": None,
#                         "body": []
#                     }

#                 elif role == "sectionHeading" and current_article:
#                     if current_article["subheading"] is None:
#                         current_article["subheading"] = text
#                     else:
#                         current_article["body"].append(text)

#                 elif current_article:
#                     current_article["body"].append(text)

#             if current_article:
#                 article_list.append(current_article)

#             return article_list
        
#         # Step 3: Process each column (EXACT Colab logic)
#         all_articles = []
        
#         for col, paras in articles_by_column.items():
#             # Sort paragraphs by Y position (top to bottom)
#             def get_top_y(p):
#                 if not p.bounding_regions:
#                     return 0
#                 box = p.bounding_regions[0].polygon
#                 return min(point.y for point in box)
            
#             sorted_paras = sorted(paras, key=get_top_y)
            
#             # Apply your exact article splitting function
#             column_articles = split_into_articles(sorted_paras)
            
#             print(f"\n📰 COLUMN: {col.upper()} - Found {len(column_articles)} articles")
            
#             for i, art in enumerate(column_articles, 1):
#                 print(f"  ARTICLE {i}")
#                 print(f"    HEADING: {art['heading'][:80] if art['heading'] else 'No heading'}...")
#                 print(f"    SUBHEADING: {art['subheading'][:80] if art['subheading'] else 'No subheading'}...")
#                 print(f"    BODY PARAGRAPHS: {len(art['body'])}")
                
#                 # Create full text for each article
#                 full_text = ""
#                 if art["heading"]:
#                     full_text += art["heading"] + "\n"
#                 if art["subheading"]:
#                     full_text += art["subheading"] + "\n"
#                 if art["body"]:
#                     full_text += "\n".join(art["body"])
                
#                 # Add column information
#                 art["column"] = col
#                 art["full_text"] = full_text
#                 art["article_number"] = i
#                 art["article_id"] = f"{col}_{i}"  # Unique ID for selection
                
#                 all_articles.append(art)
        
#         print(f"\nTotal articles found across all columns: {len(all_articles)}")
        
#         # Step 4: If no articles found with role-based detection, try alternative method
#         if len(all_articles) == 0:
#             print("No articles found with role detection. Trying alternative grouping...")
#             all_articles = alternative_article_grouping(result)
        
#         # Step 5: Sort articles by position (left column first, then right column, top to bottom)
#         def get_article_position(article):
#             # Find the first paragraph of the article to determine position
#             for p in result.paragraphs:
#                 if p.content.strip() == article.get("heading", "").strip():
#                     if p.bounding_regions:
#                         box = p.bounding_regions[0].polygon
#                         x_avg = sum(point.x for point in box) / len(box)
#                         y_avg = sum(point.y for point in box) / len(box)
#                         column = 0 if x_avg < 0.5 else 1
#                         return (column, y_avg)
#             return (0, 0)
        
#         # Sort articles by column, then by Y position
#         sorted_articles = sorted(all_articles, key=lambda a: get_article_position(a))
        
#         # Reindex articles with unique IDs
#         for i, article in enumerate(sorted_articles):
#             article["article_index"] = i + 1
#             if "article_id" not in article:
#                 article["article_id"] = f"article_{i+1}"
        
#         return sorted_articles
        
#     except Exception as e:
#         print(f"Error extracting articles: {e}")
#         import traceback
#         traceback.print_exc()
#         return []

# def alternative_article_grouping(result):
#     """
#     Alternative method for grouping paragraphs into articles when role detection fails.
#     This is still based on your Colab logic but with different heuristics.
#     """
#     try:
#         articles = []
#         current_group = []
#         last_y = -1
#         y_threshold = 0.02  # 2% of page height
        
#         # Get all paragraphs sorted by Y position
#         def get_top_y(p):
#             if not p.bounding_regions:
#                 return 0
#             box = p.bounding_regions[0].polygon
#             return min(point.y for point in box)
        
#         sorted_paragraphs = sorted(result.paragraphs, key=get_top_y)
        
#         for p in sorted_paragraphs:
#             text = p.content.strip()
#             if len(text) < 10:  # Skip very short text
#                 continue
            
#             current_y = get_top_y(p)
            
#             # Check if this should start a new article
#             is_new_article = False
            
#             # Heuristic 1: Check for heading patterns (like in Colab)
#             words = text.split()
#             if 2 <= len(words) <= 15:
#                 # Check for uppercase or ending with punctuation
#                 if (text.isupper() or 
#                     text.endswith(".") or 
#                     text.endswith(":") or
#                     text.endswith("?") or
#                     text.endswith("!")):
#                     is_new_article = True
            
#             # Heuristic 2: Large vertical gap
#             if last_y != -1 and (current_y - last_y) > y_threshold and current_group:
#                 is_new_article = True
            
#             if is_new_article and current_group:
#                 # Create article from current group
#                 heading = current_group[0] if len(current_group[0].split()) <= 15 else ""
#                 body = [p.content for p in current_group[1:]] if len(current_group) > 1 else [p.content for p in current_group]
                
#                 articles.append({
#                     "heading": heading,
#                     "subheading": None,
#                     "body": body,
#                     "full_text": "\n".join([p.content for p in current_group])
#                 })
#                 current_group = [p]
#             else:
#                 current_group.append(p)
            
#             last_y = current_y
        
#         # Add the last group
#         if current_group:
#             heading = current_group[0].content if len(current_group[0].content.split()) <= 15 else ""
#             body = [p.content for p in current_group[1:]] if len(current_group) > 1 else [p.content for p in current_group]
            
#             articles.append({
#                 "heading": heading,
#                 "subheading": None,
#                 "body": body,
#                 "full_text": "\n".join([p.content for p in current_group])
#             })
        
#         print(f"Alternative grouping found {len(articles)} articles")
        
#         # Add article IDs
#         for i, article in enumerate(articles):
#             article["article_id"] = f"alt_article_{i+1}"
        
#         return articles
        
#     except Exception as e:
#         print(f"Alternative grouping error: {e}")
#         return []

# # ============================================================
# # TEXT EXTRACTION DISPATCHER
# # ============================================================

# def extract_text_with_strategy(input_path: str, resource_type: str) -> Dict[str, Any]:
#     """
#     Extract text using appropriate strategy based on resource type.
#     """
#     print(f"\nExtracting text for resource type: {resource_type}")
    
#     if resource_type in ["newspapers", "magazine"]:
#         # Use Azure Document Intelligence for newspapers and magazines
#         azure_result = extract_with_azure(input_path)
        
#         if azure_result and azure_result.get("articles"):
#             print(f"✓ Azure extraction successful: {len(azure_result['articles'])} articles found")
            
#             # Extract article texts
#             article_texts = []
#             for article in azure_result["articles"]:
#                 article_texts.append(article.get("full_text", ""))
            
#             return {
#                 "full_text": azure_result["full_text"],
#                 "article_texts": article_texts,
#                 "method": "azure",
#                 "structured_articles": azure_result["articles"]
#             }
#         else:
#             print("⚠ Azure extraction failed or no articles found. Falling back to simple OCR...")
#             # Fallback to simple OCR
#             full_text = extract_text_book(input_path)
#             return {
#                 "full_text": full_text,
#                 "article_texts": [full_text],
#                 "method": "simple_ocr_fallback",
#                 "structured_articles": []
#             }
    
#     else:  # For books
#         # Use simple OCR for books
#         print("Using simple OCR for book content...")
#         full_text = extract_text_book(input_path)
#         return {
#             "full_text": full_text,
#             "article_texts": [full_text],
#             "method": "simple_ocr",
#             "structured_articles": []
#         }

# # ============================================================
# # RESOURCE TYPE DETECTION
# # ============================================================

# def predict_resource_type(img_path: str) -> tuple[str, float]:
#     """Predict resource type from image."""
#     try:
#         img = image.load_img(img_path, target_size=IMG_SIZE)
#         arr = image.img_to_array(img) / 255.0
#         arr = np.expand_dims(arr, 0)
#         pred = type_model.predict(arr, verbose=0)
#         cls = CLASS_NAMES[np.argmax(pred)]
#         conf = float(np.max(pred))
#         return cls.lower(), conf
#     except Exception as e:
#         print(f"Resource type detection error: {e}")
#         return "books", 0.5  # Default fallback

# # ============================================================
# # SUMMARIZATION FUNCTIONS
# # ============================================================

# def get_prefix(type_name: str) -> str:
#     """Get summarization prefix based on type."""
#     if type_name == "newspapers":
#         return "summarize: short summary: "
#     elif type_name == "magazine":
#         return "summarize: medium summary: "
#     elif type_name == "books":
#         return "summarize: long summary in detail: "
#     return "summarize: "

# def summarize_text(text: str, source_type: str) -> str:
#     """Summarize text using T5 model."""
#     try:
#         if not text.strip():
#             return "No text available for summarization."
            
#         prefix = get_prefix(source_type)
        
#         # Take first 2000 characters for summarization
#         input_text = prefix + text[:2000]
        
#         inputs = summ_tokenizer(
#             input_text,
#             return_tensors="pt",
#             max_length=1024,
#             truncation=True,
#             padding="max_length"
#         ).to(DEVICE)
        
#         with torch.no_grad():
#             output_ids = summ_model.generate(
#                 input_ids=inputs['input_ids'],
#                 attention_mask=inputs['attention_mask'],
#                 max_length=300,
#                 num_beams=4,
#                 early_stopping=True
#             )
        
#         return summ_tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     except Exception as e:
#         print(f"Summarization error: {e}")
#         # Fallback: return first 500 characters
#         return text[:500] + "..." if len(text) > 500 else text

# def summarize_specific_article(articles: List[Dict[str, Any]], article_id: str, resource_type: str) -> Dict[str, Any]:
#     """Summarize a specific article by its ID."""
#     for article in articles:
#         if article.get("article_id") == article_id or article.get("heading", "").strip() == article_id.strip():
#             print(f"Summarizing article: {article.get('heading', 'No heading')[:50]}...")
            
#             # Get full text of the article
#             article_text = article.get("full_text", "")
#             if not article_text:
#                 # Reconstruct from parts
#                 article_text = ""
#                 if article.get("heading"):
#                     article_text += article["heading"] + "\n"
#                 if article.get("subheading"):
#                     article_text += article["subheading"] + "\n"
#                 if article.get("body"):
#                     article_text += "\n".join(article["body"])
            
#             # Generate summary
#             summary = summarize_text(article_text, resource_type)
            
#             return {
#                 "article_id": article_id,
#                 "heading": article.get("heading", "No heading"),
#                 "subheading": article.get("subheading", ""),
#                 "column": article.get("column", "unknown"),
#                 "article_index": article.get("article_index", 0),
#                 "full_text_preview": article_text[:200] + "..." if len(article_text) > 200 else article_text,
#                 "summary": summary,
#                 "word_count": len(article_text.split()),
#                 "paragraph_count": len(article.get("body", [])),
#                 "timestamp": datetime.now().isoformat()
#             }
    
#     return {
#         "error": "Article not found",
#         "article_id": article_id,
#         "available_articles": [a.get("article_id") for a in articles if a.get("article_id")]
#     }

# # ============================================================
# # QUESTION ANSWERING FUNCTIONS (FIXED VERSION)
# # ============================================================

# def softmax_1d(x: torch.Tensor) -> torch.Tensor:
#     """Compute softmax for 1D tensor."""
#     x = x - x.max()
#     return torch.exp(x) / torch.exp(x).sum()

# @torch.inference_mode()
# def answer_question_with_sliding_window(
#     question: str,
#     context: str,
#     max_answer_len: int = 64,
#     score_threshold: float = 0.15,
#     max_seq_len: int = 384,
#     doc_stride: int = 128,
# ) -> Dict[str, Any]:
#     """
#     Robust extractive QA with proper padding and truncation handling.
#     """
    
#     # Handle empty context
#     if not context.strip():
#         return {
#             "answer": "No context provided to answer the question.",
#             "confidence": 0.0,
#             "context_preview": ""
#         }
    
#     # Handle very short context
#     if len(context.split()) < 5:
#         return {
#             "answer": "Context is too short to provide a meaningful answer.",
#             "confidence": 0.0,
#             "context_preview": context
#         }
    
#     try:
#         # Tokenize with proper padding and truncation
#         enc = qa_tokenizer(
#             question,
#             context,
#             truncation="only_second",
#             max_length=max_seq_len,
#             stride=doc_stride,
#             return_overflowing_tokens=True,
#             return_offsets_mapping=True,
#             padding="max_length",  # Add padding to ensure same length
#             return_tensors="pt",
#         )

#         input_ids = enc["input_ids"].to(DEVICE)
#         attention_mask = enc["attention_mask"].to(DEVICE)
#         offsets = enc["offset_mapping"]  # (num_chunks, seq_len, 2)

#         # Check if we have any chunks
#         if input_ids.size(0) == 0:
#             return {
#                 "answer": "Unable to process the context. It might be too short or malformed.",
#                 "confidence": 0.0,
#                 "context_preview": context[:200] + "..." if len(context) > 200 else context
#             }

#         best = {
#             "answer": "",
#             "score": -1.0,
#             "start_char": None,
#             "end_char": None,
#             "chunk_index": None,
#         }

#         outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
#         start_logits = outputs.start_logits  # (num_chunks, seq_len)
#         end_logits = outputs.end_logits      # (num_chunks, seq_len)

#         for ci in range(input_ids.size(0)):
#             s_logits = start_logits[ci]
#             e_logits = end_logits[ci]

#             # Convert logits to probabilities for a nicer confidence score
#             s_probs = softmax_1d(s_logits)
#             e_probs = softmax_1d(e_logits)

#             # Only allow answer tokens from the context part, not question/special tokens
#             chunk_offsets = offsets[ci]  # (seq_len, 2)

#             valid_positions = []
#             for ti, (a, b) in enumerate(chunk_offsets.tolist()):
#                 if attention_mask[ci, ti].item() == 0:
#                     continue
#                 # exclude special tokens
#                 tok_id = input_ids[ci, ti].item()
#                 if tok_id in [qa_tokenizer.cls_token_id, qa_tokenizer.sep_token_id, qa_tokenizer.pad_token_id]:
#                     continue
#                 # offsets that are (0,0) tend to be non-context in many tokenizers
#                 if a == 0 and b == 0:
#                     continue
#                 valid_positions.append(ti)

#             if not valid_positions:
#                 continue

#             # Pick top candidate spans efficiently
#             k = min(10, len(valid_positions))
#             top_starts = torch.topk(s_probs, k=k).indices.tolist()
#             top_ends = torch.topk(e_probs, k=k).indices.tolist()

#             for s in top_starts:
#                 if s not in valid_positions:
#                     continue
#                 for e in top_ends:
#                     if e not in valid_positions:
#                         continue
#                     if e < s:
#                         continue
#                     if (e - s + 1) > max_answer_len:
#                         continue

#                     score = (s_probs[s] * e_probs[e]).item()

#                     if score > best["score"]:
#                         start_char, end_char = chunk_offsets[s].tolist()[0], chunk_offsets[e].tolist()[1]
#                         # Guard: sometimes offsets can be weird; ensure valid slice
#                         if 0 <= start_char < end_char <= len(context):
#                             ans = context[start_char:end_char].strip()
#                         else:
#                             # Try to get answer from tokens
#                             ans_tokens = input_ids[ci, s:e+1]
#                             ans = qa_tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
                        
#                         if ans:  # Only update if we got a valid answer
#                             best.update(
#                                 {
#                                     "answer": ans,
#                                     "score": score,
#                                     "start_char": start_char if 0 <= start_char < end_char <= len(context) else None,
#                                     "end_char": end_char if 0 <= start_char < end_char <= len(context) else None,
#                                     "chunk_index": ci,
#                                 }
#                             )

#         # If confidence too low, return a safe fallback
#         if best["score"] < score_threshold or not best["answer"]:
#             # Try a simpler approach with direct QA
#             try:
#                 # Single chunk approach for short contexts
#                 inputs = qa_tokenizer(
#                     question,
#                     context,
#                     truncation=True,
#                     max_length=512,
#                     padding=True,
#                     return_tensors="pt"
#                 ).to(DEVICE)
                
#                 with torch.no_grad():
#                     outputs = qa_model(**inputs)
                
#                 start_idx = torch.argmax(outputs.start_logits)
#                 end_idx = torch.argmax(outputs.end_logits)
                
#                 if end_idx >= start_idx:
#                     answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
#                     answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
#                     confidence = (torch.softmax(outputs.start_logits, dim=1)[0, start_idx] * 
#                                  torch.softmax(outputs.end_logits, dim=1)[0, end_idx]).item()
                    
#                     if confidence > score_threshold and answer.strip():
#                         return {
#                             "answer": answer,
#                             "confidence": float(confidence),
#                             "context_preview": context[:200] + "..." if len(context) > 200 else context
#                         }
#             except Exception:
#                 pass
            
#             return {
#                 "answer": "I couldn't find a confident answer in the provided text. Try asking a more specific question or provide more context.",
#                 "confidence": float(best["score"] if best["score"] >= 0 else 0.0),
#                 "context_preview": context[:200] + "..." if len(context) > 200 else context,
#             }

#         return {
#             "answer": best["answer"],
#             "confidence": float(best["score"]),
#             "context_preview": context[:200] + "..." if len(context) > 200 else context,
#             "start_char": int(best["start_char"]) if best["start_char"] is not None else None,
#             "end_char": int(best["end_char"]) if best["end_char"] is not None else None,
#         }
        
#     except Exception as e:
#         print(f"Q&A Error: {e}")
#         # Fallback to simple keyword matching or return helpful message
#         return {
#             "answer": f"Error processing question: {str(e)[:100]}. Please try rephrasing your question.",
#             "confidence": 0.0,
#             "context_preview": context[:200] + "..." if len(context) > 200 else context,
#         }

# def answer_question_for_article(
#     document_id: str,
#     article_id: str,
#     question: str,
#     max_answer_len: int = 64,
#     score_threshold: float = 0.15
# ) -> Dict[str, Any]:
#     """Answer a question about a specific article with better error handling."""
    
#     if document_id not in processed_documents:
#         return {
#             "error": "Document not found",
#             "document_id": document_id,
#             "suggestion": "Please process the document first using /process endpoint"
#         }
    
#     document_data = processed_documents[document_id]
#     resource_type = document_data["resource_type"]
#     structured_articles = document_data.get("structured_articles", [])
#     full_text = document_data["full_text"]
    
#     # Validate question
#     if not question or len(question.strip()) < 3:
#         return {
#             "error": "Question is too short. Please ask a more specific question.",
#             "document_id": document_id,
#             "article_id": article_id
#         }
    
#     # Find the article text
#     article_text = ""
#     article_heading = ""
    
#     if not structured_articles:
#         # For books or documents without structured articles
#         if article_id == "full_document":
#             article_text = full_text
#             article_heading = "Full Document"
#         else:
#             return {
#                 "error": "Article not found",
#                 "article_id": article_id,
#                 "suggestion": "Use 'full_document' as article_id for this document",
#                 "available_articles": ["full_document"]
#             }
#     else:
#         # Find the specific article
#         article_found = False
#         for article in structured_articles:
#             if article.get("article_id") == article_id:
#                 article_text = article.get("full_text", "")
#                 article_heading = article.get("heading", "Unknown Article")
#                 article_found = True
#                 break
        
#         if not article_found:
#             available = [a.get("article_id") for a in structured_articles if a.get("article_id")]
#             return {
#                 "error": "Article not found",
#                 "article_id": article_id,
#                 "available_articles": available[:10]  # Limit to 10 for display
#             }
    
#     if not article_text.strip():
#         return {
#             "error": "Article text is empty",
#             "article_id": article_id,
#             "article_heading": article_heading,
#             "suggestion": "Try a different article or reprocess the document"
#         }
    
#     # Clean the text - remove excessive whitespace and special characters
#     article_text = re.sub(r'\s+', ' ', article_text).strip()
    
#     # Check if context is too short
#     if len(article_text.split()) < 10:
#         return {
#             "document_id": document_id,
#             "article_id": article_id,
#             "article_heading": article_heading,
#             "question": question,
#             "answer": "The article text is too short to answer questions. Please select a longer article.",
#             "confidence": 0.0,
#             "context_preview": article_text,
#             "resource_type": resource_type,
#             "timestamp": datetime.now().isoformat()
#         }
    
#     print(f"Q&A - Context length: {len(article_text)} chars, {len(article_text.split())} words")
#     print(f"Q&A - Question: {question}")
    
#     # Get answer using Q&A model
#     qa_result = answer_question_with_sliding_window(
#         question=question,
#         context=article_text,
#         max_answer_len=max_answer_len,
#         score_threshold=score_threshold
#     )
    
#     # Prepare response
#     response = {
#         "document_id": document_id,
#         "article_id": article_id,
#         "article_heading": article_heading,
#         "question": question,
#         "answer": qa_result["answer"],
#         "confidence": qa_result["confidence"],
#         "context_preview": qa_result.get("context_preview", article_text[:200] + "..." if len(article_text) > 200 else article_text),
#         "resource_type": resource_type,
#         "timestamp": datetime.now().isoformat()
#     }
    
#     # Add debug info if confidence is low
#     if qa_result["confidence"] < 0.3:
#         response["suggestion"] = "Try asking a more specific question or selecting a different article"
    
#     return response



# # ============================================================
# # MAIN PROCESSING
# # ============================================================

# # Global variable to store processed documents for article selection
# processed_documents = {}

# def process_document(input_path: str) -> dict:
#     """Process document through full pipeline."""
#     try:
#         print("\n" + "="*60)
#         print("STARTING DOCUMENT PROCESSING")
#         print("="*60)
        
#         # Step 1: Detect resource type
#         print("\n[1/4] Detecting resource type...")
#         if input_path.lower().endswith(".pdf"):
#             pages = convert_from_path(input_path, first_page=1, last_page=1)
#             if pages:
#                 temp_dir = tempfile.gettempdir()
#                 temp_img = os.path.join(temp_dir, f"temp_detect_{os.urandom(4).hex()}.jpg")
#                 pages[0].save(temp_img, "JPEG", quality=95)
#                 resource_type, conf = predict_resource_type(temp_img)
#                 try:
#                     os.remove(temp_img)
#                 except:
#                     pass
#             else:
#                 resource_type, conf = "books", 0.5
#         else:
#             resource_type, conf = predict_resource_type(input_path)
        
#         print(f"✓ Resource type: {resource_type} (confidence: {conf:.2f})")
        
#         # Step 2: Extract text with appropriate strategy
#         print("\n[2/4] Extracting text with optimized strategy...")
#         extraction_result = extract_text_with_strategy(input_path, resource_type)
#         full_text = extraction_result["full_text"]
#         article_texts = extraction_result["article_texts"]
#         extraction_method = extraction_result["method"]
#         structured_articles = extraction_result.get("structured_articles", [])
        
#         if not full_text or len(full_text.strip()) == 0:
#             raise ValueError("No text extracted from document")
        
#         print(f"✓ Extracted {len(full_text)} characters using {extraction_method}")
#         print(f"✓ Found {len(article_texts)} article(s)")
        
#         # Step 3: Prepare structured article data if available
#         article_list_for_selection = []
#         if structured_articles:
#             for i, article in enumerate(structured_articles[:10]):  # Limit to 10 articles for display
#                 article_list_for_selection.append({
#                     "index": i + 1,
#                     "article_id": article.get("article_id", f"article_{i+1}"),
#                     "column": article.get("column", "unknown"),
#                     "heading": article.get("heading", "No heading") or "No heading",
#                     "subheading": article.get("subheading", "") or "",
#                     "body_preview": " ".join(article.get("body", [])[:2])[:150] + "..." if article.get("body") else "",
#                     "word_count": len(article.get("full_text", "").split()),
#                     "paragraph_count": len(article.get("body", [])),
#                     "is_main_article": i == 0  # First article is considered main
#                 })
#         else:
#             # Create dummy article for books or when no structure detected
#             article_list_for_selection.append({
#                 "index": 1,
#                 "article_id": "full_document",
#                 "column": "full",
#                 "heading": "Full Document",
#                 "subheading": "Complete extracted text",
#                 "body_preview": full_text[:150] + "..." if len(full_text) > 150 else full_text,
#                 "word_count": len(full_text.split()),
#                 "paragraph_count": len(full_text.split('\n\n')),
#                 "is_main_article": True
#             })
        
#         # Step 4: Generate ONE summary for the main article only (not all articles)
#         print("\n[3/4] Generating summary for main article...")
#         summaries = []
        
#         # Only summarize the first/main article initially
#         if article_list_for_selection:
#             main_article_id = article_list_for_selection[0]["article_id"]
            
#             if structured_articles:
#                 # Find the main article in structured articles
#                 main_article_text = ""
#                 for article in structured_articles:
#                     if article.get("article_id") == main_article_id:
#                         main_article_text = article.get("full_text", "")
#                         break
                
#                 if not main_article_text and structured_articles:
#                     main_article_text = structured_articles[0].get("full_text", "")
#             else:
#                 main_article_text = full_text
            
#             if main_article_text.strip():
#                 print(f"  Summarizing main article: {article_list_for_selection[0]['heading'][:50]}...")
#                 summary = summarize_text(main_article_text, resource_type)
#                 summaries.append({
#                     "article_id": main_article_id,
#                     "heading": article_list_for_selection[0]["heading"],
#                     "summary": summary,
#                     "is_main": True
#                 })
#             else:
#                 summaries.append({
#                     "article_id": main_article_id,
#                     "heading": article_list_for_selection[0]["heading"],
#                     "summary": "Could not generate summary for this article.",
#                     "is_main": True
#                 })
        
#         print(f"✓ Generated {len(summaries)} summary for main article")
        
#         # Create a document ID for later reference
#         import hashlib
#         document_id = hashlib.md5(f"{input_path}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        
#         # Store structured articles for later selective summarization AND Q&A
#         processed_documents[document_id] = {
#             "resource_type": resource_type,
#             "structured_articles": structured_articles,
#             "full_text": full_text,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         result = {
#             "document_id": document_id,
#             "resource_type": resource_type,
#             "confidence": conf,
#             "extracted_text_preview": full_text[:500] + "..." if len(full_text) > 500 else full_text,
#             "article_list": article_list_for_selection,
#             "summaries": summaries,  # Only main article summary
#             "num_articles": len(article_list_for_selection),
#             "text_length": len(full_text),
#             "extraction_method": extraction_method,
#             "processing_info": {
#                 "ocr_method": extraction_method,
#                 "article_detection": extraction_method == "azure",
#                 "supports_selective_summarization": len(structured_articles) > 0,
#                 "supports_qa": True,  # NEW: All documents now support Q&A
#                 "timestamp": datetime.now().isoformat()
#             }
#         }
        
#         print("\n" + "="*60)
#         print("PROCESSING COMPLETE")
#         print(f"Document ID: {document_id}")
#         print("Use /summarize-article endpoint to get summaries for specific articles")
#         print("Use /ask-question endpoint to ask questions about any article")  # NEW
#         print("="*60 + "\n")
        
#         return result
        
#     except Exception as e:
#         print(f"\n❌ Processing error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise Exception(f"Processing error: {str(e)}")

# # ============================================================
# # FASTAPI APPLICATION
# # ============================================================

# app = FastAPI(title="EXACT Colab Article Detection Document Processor with Q&A", version="6.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     """Health check endpoint."""
#     return {
#         "status": "ok",
#         "message": "EXACT Colab Article Detection Document Processor API v6.0 with Q&A",
#         "features": {
#             "simple_ocr": True,
#             "azure_document_intelligence": True,
#             "resource_type_detection": True,
#             "summarization": True,
#             "exact_colab_article_detection": True,
#             "column_based_article_grouping": True,
#             "selective_article_summarization": True,
#             "question_answering": True  # NEW FEATURE
#         },
#         "models_loaded": {
#             "resource_type": type_model is not None,
#             "summarization": summ_model is not None,
#             "question_answering": qa_model is not None
#         },
#         "qa_model": QA_MODEL_NAME,
#         "azure_configured": bool(AZURE_ENDPOINT and AZURE_KEY),
#         "endpoints": {
#             "POST /process": "Process a document and extract articles",
#             "POST /summarize-article": "Summarize a specific article by ID or heading",
#             "POST /ask-question": "Ask questions about a specific article",  # NEW
#             "GET /articles/{document_id}": "Get article list for a processed document",
#             "GET /cleanup": "Clean up old documents"
#         }
#     }

# @app.post("/process")
# async def process_file(file: UploadFile = File(...)):
#     """Process uploaded document."""
#     allowed_types = [
#         "application/pdf",
#         "image/jpeg",
#         "image/jpg",
#         "image/png",
#         "image/tiff",
#         "image/bmp"
#     ]
    
#     if file.content_type not in allowed_types:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file type: {file.content_type}"
#         )
    
#     temp_dir = tempfile.gettempdir()
#     file_ext = os.path.splitext(file.filename)[1] or (".pdf" if file.content_type == "application/pdf" else ".jpg")
#     temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{file_ext}")
    
#     try:
#         with open(temp_path, "wb") as f:
#             content = await file.read()
#             f.write(content)
        
#         print(f"\n{'='*60}")
#         print(f"Processing uploaded file: {file.filename}")
#         print(f"{'='*60}")
        
#         result = process_document(temp_path)
        
#         return JSONResponse(content=result)
        
#     except Exception as e:
#         print(f"\n❌ Processing failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
#     finally:
#         if os.path.exists(temp_path):
#             try:
#                 os.remove(temp_path)
#             except:
#                 pass

# @app.post("/summarize-article")
# async def summarize_specific_article_endpoint(
#     document_id: str = Form(...),
#     article_id: str = Form(...)
# ):
#     """Summarize a specific article from a previously processed document."""
    
#     if document_id not in processed_documents:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Document ID {document_id} not found. Please process the document first."
#         )
    
#     document_data = processed_documents[document_id]
#     resource_type = document_data["resource_type"]
#     structured_articles = document_data.get("structured_articles", [])
#     full_text = document_data["full_text"]
    
#     print(f"\n{'='*60}")
#     print(f"Summarizing article {article_id} from document {document_id}")
#     print(f"{'='*60}")
    
#     if not structured_articles:
#         # For books or documents without structured articles
#         if article_id == "full_document":
#             # Summarize the full document
#             summary = summarize_text(full_text, resource_type)
#             return {
#                 "document_id": document_id,
#                 "article_id": article_id,
#                 "heading": "Full Document",
#                 "summary": summary,
#                 "word_count": len(full_text.split()),
#                 "full_text_preview": full_text[:300] + "..." if len(full_text) > 300 else full_text,
#                 "resource_type": resource_type,
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail="This document does not have structured articles. Use 'full_document' as article_id."
#             )
    
#     # Find and summarize the specific article
#     result = summarize_specific_article(structured_articles, article_id, resource_type)
    
#     if "error" in result:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Article '{article_id}' not found. Available articles: {result.get('available_articles', [])}"
#         )
    
#     # Add document info
#     result["document_id"] = document_id
#     result["resource_type"] = resource_type
    
#     return JSONResponse(content=result)

# @app.post("/ask-question")
# async def ask_question_endpoint(
#     request: QARequest
# ):
#     """Ask a question about a specific article with better error handling."""
    
#     print(f"\n{'='*60}")
#     print(f"Question about document {request.document_id}, article {request.article_id}")
#     print(f"Question: {request.question}")
#     print(f"Max answer length: {request.max_answer_len}")
#     print(f"Score threshold: {request.score_threshold}")
#     print(f"{'='*60}")
    
#     try:
#         result = answer_question_for_article(
#             document_id=request.document_id,
#             article_id=request.article_id,
#             question=request.question,
#             max_answer_len=request.max_answer_len,
#             score_threshold=request.score_threshold
#         )
        
#         if "error" in result:
#             status_code = 404 if "not found" in result["error"].lower() else 400
#             raise HTTPException(
#                 status_code=status_code,
#                 detail=result["error"]
#             )
        
#         return JSONResponse(content=result)
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Unexpected error in Q&A: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error: {str(e)[:100]}"
#         )

# @app.get("/articles/{document_id}")
# async def get_articles_list(document_id: str):
#     """Get the list of articles for a previously processed document."""
    
#     if document_id not in processed_documents:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Document ID {document_id} not found."
#         )
    
#     document_data = processed_documents[document_id]
#     structured_articles = document_data.get("structured_articles", [])
    
#     article_list = []
#     if structured_articles:
#         for i, article in enumerate(structured_articles[:20]):  # Limit to 20 articles
#             article_list.append({
#                 "index": i + 1,
#                 "article_id": article.get("article_id", f"article_{i+1}"),
#                 "column": article.get("column", "unknown"),
#                 "heading": article.get("heading", "No heading") or "No heading",
#                 "subheading": article.get("subheading", "") or "",
#                 "body_preview": " ".join(article.get("body", [])[:2])[:150] + "..." if article.get("body") else "",
#                 "word_count": len(article.get("full_text", "").split()),
#                 "paragraph_count": len(article.get("body", []))
#             })
#     else:
#         # For books or documents without structured articles
#         article_list.append({
#             "index": 1,
#             "article_id": "full_document",
#             "column": "full",
#             "heading": "Full Document",
#             "subheading": "Complete extracted text",
#             "body_preview": document_data["full_text"][:150] + "..." if len(document_data["full_text"]) > 150 else document_data["full_text"],
#             "word_count": len(document_data["full_text"].split()),
#             "paragraph_count": len(document_data["full_text"].split('\n\n'))
#         })
    
#     return {
#         "document_id": document_id,
#         "resource_type": document_data["resource_type"],
#         "num_articles": len(article_list),
#         "articles": article_list,
#         "timestamp": document_data["timestamp"],
#         "supports_qa": True  # All documents support Q&A
#     }

# @app.get("/cleanup")
# async def cleanup_old_documents(hours: int = 24):
#     """Clean up old processed documents."""
#     cutoff_time = datetime.now().timestamp() - (hours * 3600)
#     removed_count = 0
    
#     for doc_id in list(processed_documents.keys()):
#         try:
#             doc_time = datetime.fromisoformat(processed_documents[doc_id]["timestamp"]).timestamp()
#             if doc_time < cutoff_time:
#                 del processed_documents[doc_id]
#                 removed_count += 1
#         except:
#             # If timestamp parsing fails, remove the document
#             del processed_documents[doc_id]
#             removed_count += 1
    
#     return {
#         "message": f"Cleaned up {removed_count} old documents",
#         "remaining_documents": len(processed_documents),
#         "cleanup_hours": hours
#     }

# # Install Azure packages if needed
# def check_and_install_azure_packages():
#     """Check if Azure packages are installed, install if not."""
#     try:
#         import azure.ai.formrecognizer
#         import azure.core
#         print("✓ Azure packages already installed")
#     except ImportError:
#         print("⚠ Azure packages not found. Installing...")
#         import subprocess
#         import sys
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "azure-ai-formrecognizer", "azure-core"])
#         print("✓ Azure packages installed")

# if __name__ == "__main__":
#     # Check and install Azure packages if needed
#     check_and_install_azure_packages()
    
#     print("\n" + "="*60)
#     print("STARTING EXACT COLAB ARTICLE DETECTION PROCESSOR WITH Q&A")
#     print("="*60 + "\n")
#     print("NEW FEATURES:")
#     print("  1. Users can select specific articles to summarize!")
#     print("  2. Users can ask questions about any article!")
#     print("\nEndpoints available:")
#     print("  POST /process - Upload and process a document")
#     print("  POST /summarize-article - Summarize a specific article")
#     print("  POST /ask-question - Ask questions about an article")
#     print("  GET /articles/{document_id} - Get article list")
#     print("  GET /cleanup - Clean up old documents")
#     print("="*60 + "\n")
    
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# main.py
"""
Fixed Article Detection for Document Processing
With EXACT Colab logic for article detection
Added: Selective article summarization
Added: Question Answering system
"""

import os
import sys
import pytesseract
from pathlib import Path

# Add Tesseract configuration from config
from config import BASE_DIR

# Set Tesseract path for Windows
if sys.platform == "win32":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✓ Tesseract found at: {path}")
            break
    else:
        print("⚠ Warning: Tesseract not found. Please install Tesseract-OCR")

import uvicorn
from api_server import app, check_and_install_azure_packages

if __name__ == "__main__":
    # Check and install Azure packages if needed
    check_and_install_azure_packages()
    
    print("\n" + "="*60)
    print("STARTING EXACT COLAB ARTICLE DETECTION PROCESSOR WITH Q&A")
    print("="*60 + "\n")
    print("NEW FEATURES:")
    print("  1. Users can select specific articles to summarize!")
    print("  2. Users can ask questions about any article!")
    print("\nEndpoints available:")
    print("  POST /process - Upload and process a document")
    print("  POST /summarize-article - Summarize a specific article")
    print("  POST /ask-question - Ask questions about an article")
    print("  GET /articles/{document_id} - Get article list")
    print("  GET /cleanup - Clean up old documents")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)