# qa_system.py
import torch
import re
from typing import Dict, Any, Optional  # ADD Optional here
from datetime import datetime
from models import DEVICE

def softmax_1d(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax for 1D tensor."""
    x = x - x.max()
    return torch.exp(x) / torch.exp(x).sum()

@torch.inference_mode()
def answer_question_with_sliding_window(
    question: str,
    context: str,
    qa_tokenizer,
    qa_model,
    max_answer_len: int = 64,
    score_threshold: float = 0.15,
    max_seq_len: int = 384,
    doc_stride: int = 128,
) -> Dict[str, Any]:
    """
    Robust extractive QA with proper padding and truncation handling.
    """
    
    # Handle empty context
    if not context.strip():
        return {
            "answer": "No context provided to answer the question.",
            "confidence": 0.0,
            "context_preview": ""
        }
    
    # Handle very short context
    if len(context.split()) < 5:
        return {
            "answer": "Context is too short to provide a meaningful answer.",
            "confidence": 0.0,
            "context_preview": context
        }
    
    try:
        # Tokenize with proper padding and truncation
        enc = qa_tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",  # Add padding to ensure same length
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        offsets = enc["offset_mapping"]  # (num_chunks, seq_len, 2)

        # Check if we have any chunks
        if input_ids.size(0) == 0:
            return {
                "answer": "Unable to process the context. It might be too short or malformed.",
                "confidence": 0.0,
                "context_preview": context[:200] + "..." if len(context) > 200 else context
            }

        best = {
            "answer": "",
            "score": -1.0,
            "start_char": None,
            "end_char": None,
            "chunk_index": None,
        }

        outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits  # (num_chunks, seq_len)
        end_logits = outputs.end_logits      # (num_chunks, seq_len)

        for ci in range(input_ids.size(0)):
            s_logits = start_logits[ci]
            e_logits = end_logits[ci]

            # Convert logits to probabilities for a nicer confidence score
            s_probs = softmax_1d(s_logits)
            e_probs = softmax_1d(e_logits)

            # Only allow answer tokens from the context part, not question/special tokens
            chunk_offsets = offsets[ci]  # (seq_len, 2)

            valid_positions = []
            for ti, (a, b) in enumerate(chunk_offsets.tolist()):
                if attention_mask[ci, ti].item() == 0:
                    continue
                # exclude special tokens
                tok_id = input_ids[ci, ti].item()
                if tok_id in [qa_tokenizer.cls_token_id, qa_tokenizer.sep_token_id, qa_tokenizer.pad_token_id]:
                    continue
                # offsets that are (0,0) tend to be non-context in many tokenizers
                if a == 0 and b == 0:
                    continue
                valid_positions.append(ti)

            if not valid_positions:
                continue

            # Pick top candidate spans efficiently
            k = min(10, len(valid_positions))
            top_starts = torch.topk(s_probs, k=k).indices.tolist()
            top_ends = torch.topk(e_probs, k=k).indices.tolist()

            for s in top_starts:
                if s not in valid_positions:
                    continue
                for e in top_ends:
                    if e not in valid_positions:
                        continue
                    if e < s:
                        continue
                    if (e - s + 1) > max_answer_len:
                        continue

                    score = (s_probs[s] * e_probs[e]).item()

                    if score > best["score"]:
                        start_char, end_char = chunk_offsets[s].tolist()[0], chunk_offsets[e].tolist()[1]
                        # Guard: sometimes offsets can be weird; ensure valid slice
                        if 0 <= start_char < end_char <= len(context):
                            ans = context[start_char:end_char].strip()
                        else:
                            # Try to get answer from tokens
                            ans_tokens = input_ids[ci, s:e+1]
                            ans = qa_tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
                        
                        if ans:  # Only update if we got a valid answer
                            best.update(
                                {
                                    "answer": ans,
                                    "score": score,
                                    "start_char": start_char if 0 <= start_char < end_char <= len(context) else None,
                                    "end_char": end_char if 0 <= start_char < end_char <= len(context) else None,
                                    "chunk_index": ci,
                                }
                            )

        # If confidence too low, return a safe fallback
        if best["score"] < score_threshold or not best["answer"]:
            # Try a simpler approach with direct QA
            try:
                # Single chunk approach for short contexts
                inputs = qa_tokenizer(
                    question,
                    context,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = qa_model(**inputs)
                
                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)
                
                if end_idx >= start_idx:
                    answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
                    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    confidence = (torch.softmax(outputs.start_logits, dim=1)[0, start_idx] * 
                                 torch.softmax(outputs.end_logits, dim=1)[0, end_idx]).item()
                    
                    if confidence > score_threshold and answer.strip():
                        return {
                            "answer": answer,
                            "confidence": float(confidence),
                            "context_preview": context[:200] + "..." if len(context) > 200 else context
                        }
            except Exception:
                pass
            
            return {
                "answer": "I couldn't find a confident answer in the provided text. Try asking a more specific question or provide more context.",
                "confidence": float(best["score"] if best["score"] >= 0 else 0.0),
                "context_preview": context[:200] + "..." if len(context) > 200 else context,
            }

        return {
            "answer": best["answer"],
            "confidence": float(best["score"]),
            "context_preview": context[:200] + "..." if len(context) > 200 else context,
            "start_char": int(best["start_char"]) if best["start_char"] is not None else None,
            "end_char": int(best["end_char"]) if best["end_char"] is not None else None,
        }
        
    except Exception as e:
        print(f"Q&A Error: {e}")
        # Fallback to simple keyword matching or return helpful message
        return {
            "answer": f"Error processing question: {str(e)[:100]}. Please try rephrasing your question.",
            "confidence": 0.0,
            "context_preview": context[:200] + "..." if len(context) > 200 else context,
        }

def answer_question_for_article(
    document_id: str,
    article_id: str,
    question: str,
    processed_documents: Dict[str, Any],
    qa_tokenizer,
    qa_model,
    max_answer_len: int = 64,
    score_threshold: float = 0.15
) -> Dict[str, Any]:
    """Answer a question about a specific article with better error handling."""
    
    if document_id not in processed_documents:
        return {
            "error": "Document not found",
            "document_id": document_id,
            "suggestion": "Please process the document first using /process endpoint"
        }
    
    document_data = processed_documents[document_id]
    resource_type = document_data["resource_type"]
    structured_articles = document_data.get("structured_articles", [])
    full_text = document_data["full_text"]
    
    # Validate question
    if not question or len(question.strip()) < 3:
        return {
            "error": "Question is too short. Please ask a more specific question.",
            "document_id": document_id,
            "article_id": article_id
        }
    
    # Find the article text
    article_text = ""
    article_heading = ""
    
    if not structured_articles:
        # For books or documents without structured articles
        if article_id == "full_document":
            article_text = full_text
            article_heading = "Full Document"
        else:
            return {
                "error": "Article not found",
                "article_id": article_id,
                "suggestion": "Use 'full_document' as article_id for this document",
                "available_articles": ["full_document"]
            }
    else:
        # Find the specific article
        article_found = False
        for article in structured_articles:
            if article.get("article_id") == article_id:
                article_text = article.get("full_text", "")
                article_heading = article.get("heading", "Unknown Article")
                article_found = True
                break
        
        if not article_found:
            available = [a.get("article_id") for a in structured_articles if a.get("article_id")]
            return {
                "error": "Article not found",
                "article_id": article_id,
                "available_articles": available[:10]  # Limit to 10 for display
            }
    
    if not article_text.strip():
        return {
            "error": "Article text is empty",
            "article_id": article_id,
            "article_heading": article_heading,
            "suggestion": "Try a different article or reprocess the document"
        }
    
    # Clean the text - remove excessive whitespace and special characters
    article_text = re.sub(r'\s+', ' ', article_text).strip()
    
    # Check if context is too short
    if len(article_text.split()) < 10:
        return {
            "document_id": document_id,
            "article_id": article_id,
            "article_heading": article_heading,
            "question": question,
            "answer": "The article text is too short to answer questions. Please select a longer article.",
            "confidence": 0.0,
            "context_preview": article_text,
            "resource_type": resource_type,
            "timestamp": datetime.now().isoformat()
        }
    
    print(f"Q&A - Context length: {len(article_text)} chars, {len(article_text.split())} words")
    print(f"Q&A - Question: {question}")
    
    # Get answer using Q&A model
    qa_result = answer_question_with_sliding_window(
        question=question,
        context=article_text,
        qa_tokenizer=qa_tokenizer,
        qa_model=qa_model,
        max_answer_len=max_answer_len,
        score_threshold=score_threshold
    )
    
    # Prepare response
    response = {
        "document_id": document_id,
        "article_id": article_id,
        "article_heading": article_heading,
        "question": question,
        "answer": qa_result["answer"],
        "confidence": qa_result["confidence"],
        "context_preview": qa_result.get("context_preview", article_text[:200] + "..." if len(article_text) > 200 else article_text),
        "resource_type": resource_type,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add debug info if confidence is low
    if qa_result["confidence"] < 0.3:
        response["suggestion"] = "Try asking a more specific question or selecting a different article"
    
    return response